"""Identity and pixel-dedup checks without model execution or mass archive extraction."""

import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image,ImageOps,PngImagePlugin

from select_unseen_archive_photo import canonical_id,pixel_digests,archive_inventory


class UnseenArchiveTests(unittest.TestCase):
    def test_source_aliases_ignore_case_punctuation_and_underscores(self):
        self.assertEqual(canonical_id('Source_A-001'),canonical_id('SOURCE A 001'))
        self.assertNotEqual(canonical_id('Source_A-001'),canonical_id('Source_A-002'))

    def test_metadata_changes_do_not_hide_identical_decoded_pixels(self):
        image=Image.new('RGB',(7,5),(40,90,160));first=io.BytesIO();second=io.BytesIO()
        image.save(first,format='PNG');info=PngImagePlugin.PngInfo();info.add_text('Comment','different file encoding')
        image.save(second,format='PNG',pnginfo=info)
        self.assertNotEqual(hashlib.sha256(first.getvalue()).hexdigest(),hashlib.sha256(second.getvalue()).hexdigest())
        self.assertEqual(pixel_digests(io.BytesIO(first.getvalue())),pixel_digests(io.BytesIO(second.getvalue())))

    def test_exif_rotation_has_a_comparable_oriented_digest(self):
        image=Image.new('RGB',(7,5),(50,100,160));exif=Image.Exif();exif[274]=6
        encoded=io.BytesIO();image.save(encoded,format='JPEG',exif=exif)
        result=pixel_digests(io.BytesIO(encoded.getvalue()))
        with Image.open(io.BytesIO(encoded.getvalue())) as original:
            oriented=ImageOps.exif_transpose(original);rendered=io.BytesIO();oriented.save(rendered,format='PNG')
        reference=pixel_digests(io.BytesIO(rendered.getvalue()))
        self.assertEqual(result['exif_oriented'],reference['native'])
        self.assertEqual(result['native']['dimensions'],[7,5])
        self.assertEqual(result['exif_oriented']['dimensions'],[5,7])

    def test_archive_listing_does_not_extract_members(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root=Path(directory);path=root/'input.zip'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('folder/SOURCE_A_001.jpg',b'not-decoded-for-listing')
                archive.writestr('folder/notes.csv','a,b\n1,2\n')
            inventory,photos=archive_inventory(root)
            self.assertEqual(len(inventory),1);self.assertEqual(len(photos),1)
            self.assertEqual(photos[0]['source_id'],'SOURCE_A_001')
            self.assertEqual(list(root.iterdir()),[path])


if __name__=='__main__':unittest.main()
