# V3: ograničeni pilot adaptivnog paddinga

**Galerija usporedbe:** http://127.0.0.1:8772/

Ovo je read-only pilot za sedam slučajeva, a ne novi pregled svih 779.
V2 server na portu 8771 zaustavljen je prije pripreme pilota.

## Sačuvane odluke

Backup zaustavljenog v2:

`outputs/ruler_review_v2_backup_before_v3_20261103T154956315687Z/`

Sadrži 318 događaja, pripadajuće JSON revizije te prethodne snapshotove i dopune.
U trenutku zaustavljanja bilo je 57 dovršenih slučajeva: 32 prihvaćena,
6 ručno ispravljenih, 19 nejasnih/odrezanih; 722 slučaja bila su pending.
Broj `DECISION` događaja (59) nije isto što i broj trenutačno dovršenih slučajeva,
jer povijest uključuje prethodne odluke i revizije.

Izvorni v2 database file SHA-256 nakon zaustavljanja:
`ee38bb84d334bcf4f5b017027a5c50cc3b7fd92d12f81279eea225c8a229f375`.
Pilot samo čita postojeće izvore. Ne stvara review bazu niti prenosi odluke.

## Obuhvat i padding

Svi pilot-slučajevi zadržavaju source ID, detection ID i train ulogu:

- `C_1_2_Collembola005`: **43, 185, 219, 65, 1259**.
- `C_1_3_Collembola006`: **3, 72**.

43/185/219 izričito su traženi. 65 ima spremljenu bilješku o premalom cropu;
1259, 3 i 72 imaju raniju ljudsku dijagnozu `TRUNCATED`.

Za svaku od tri nove varijante:

```text
padding = min(64, max(8, ceil(f × max(old_width, old_height))))
f ∈ {0.10, 0.20, 0.30}
```

Padding se dodaje sa svake strane starog cropa i ograničava samo rubom originalne
fotografije. Cropovi su iz originalnih full-resolution fotografija, bez resizea
pohranjenih piksela. Stara regija unutar svakog novog cropa provjerena je kao
pixel-identical. Galerija prikazuje i zasebni širi kontekst i cijelu fotografiju.
Ti kontekstni prikazi ne idu modelu.

| Detection | Stari crop | 10% | 20% | 30% |
| --- | --- | --- | --- | --- |
| 43 | 136 × 106 | +14 px | +28 px | +41 px |
| 185 | 135 × 169 | +17 px | +34 px | +51 px |
| 219 | 89 × 169 | +17 px | +34 px | +51 px |

Usporedni stupci su stari crop, 10%, 20%, 30%, te v2 +220 px. **Svih pet varijanti
ponovno je obrađeno pri stvarnom `imgsz=320`**, s oba fiksirana modela, ukupno
70 poziva. Napomena o ranijoj provenanciji: arhivirana v2 inferenca zapravo je
koristila ulaz ≥640 prema dinamičkoj politici, unatoč naslijeđenom polju
`imgsz=320`. Ovaj pilot zato ne miješa stare v2 predikcije s novima pri 320.

## Instance i povezivanje cilja

Ultralytics vraća `retina_masks=True`; svaka binarna maska uzima se zasebno iz
`result.masks.data[i]` u izvornim pikselnim koordinatama cropa. Ne rasterizira se
`masks.xy`, ne spajaju se instance i ne odabire se najveća instanca. Spremljen je
PNG svake instance, njezin confidence, komponente, konture i geometrijska
dijagnostika. Višekomponentna maska se ne popravlja odabirom najveće komponente.

Ciljni kandidat mora imati:

- centroid maske i središte njezina bounding boxa unutar izvornog detector cropa;
- mask-vs-detector-box IoU ≥0.20;
- najmanje 60% maske unutar izvornog okvira;
- presjek koji pokriva barem 15% izvornog okvira;
- bounding-box površinu najviše dvostruko veću od izvornog okvira;
- confidence ≥0.25, jednu povezanu komponentu i bez dodira ruba cropa.

Među takvim kandidatima rangira se IoU, zatim presjek i confidence. Dva gotovo
jednako uvjerljiva kandidata (drugoplasirani IoU ≥85% najboljeg) ostavljaju slučaj
bez odabrane konture. Bez pouzdane veze `target_instance=null`, `proposal=[]`.
Pragovi su zamrznuti prije pilot-inference; nisu naučeni iz novih GT oznaka.

**Sve pilot-status oznake ostaju `PENDING`.** Zelena kontura u galeriji znači
samo geometrijski povezan kandidat, ne potvrđeno cijelo tijelo ili GT. Druge boje
prikazuju odvojene neprihvaćene instance radi dijagnoze. Slike se za prikaz
skaliraju u jednake panele; inference-cropovi i maske ostaju u native koordinatama.

## Nalaz prije moguće šire obrade

Umjereni padding zadržava cilj velikim. Na detectionu 185 izvorni bbox pri 30%
paddinga zauzima približno 159 × 200 piksela segmentacijskog ulaza, nasuprot
71 × 89 pri +220 px. Za 43 i 219 varijante 10–20% daju obećavajuće izolirane
kandidate, ali to nije dokaz točne konture.

Pilot **nije potvrdio sve uvjete na svim teškim slučajevima**:

- 185: preklopljena tijela i dvosmislena granica ostaju problem; 30% daje više
  prostora, ali geometrijski kandidat nije automatski sigurna jedna cijela jedinka.
- 65: maske i dalje dotiču rub ili imaju dodatne komponente; nema pouzdanog cilja.
- 1259: stari bbox je fragment veličine 24 × 63 px; umjereno proširenje samo po
  sebi ne potvrđuje cijelu jedinku, čak ni kada maska dobro preklapa taj fragment.
- Druga fotografija / 3: zbijena tijela i spojene/višekomponentne predikcije.
- Druga fotografija / 72: više prostora uz tijelo, ali razlike granice uz ravnalo.

Zato nije odabran univerzalni padding, `all_779_processing_allowed=false`, a
obrada/pregled svih 779 nije pokrenut. Usporedba je pripremljena za sljedeću
korisničku procjenu. Nijedna nova model-mask nije pretvorena u ručnu odluku.

## Datoteke i provjere

Pilot: `outputs/ruler_adaptive_padding_v3_pilot/`.

- `index.html`, `comparisons/*.jpg`: usporedba po slučaju.
- `context/*.jpg`: širi i full-photo kontekst, odvojen od ulaza za inferencu.
- `config.json`, `variants.json`, `pilot_cases.json`: zamrznuta geometrija i pravila.
- `manifest.json`, `predictions/`: svih 70 rezultata i svaka maska zasebno.
- `visual_assessment.json`: odvojena vizualna dijagnoza, nije GT.
- `pilot_summary.json`: brojevi geometrijskih kandidata i zatvoren full-run gate.
- `frozen_inputs/*seal.json`: hashovi izvora, modela, cropova i predikcija.

```bash
python3 -m unittest discover -s scripts/ruler_v7 -p test_adaptive_ruler_pilot.py -v
```

Šest testova provjerava adaptivnu geometriju, susjeda s većom maskom/confidenceom,
središte izvan detekcije, spojene/višekomponentne maske, dvosmislene kandidate i
dodir ruba. `pilot_adaptive_ruler_review.py` izričito dopušta samo navedenih sedam
slučajeva i nema način za obradu cijelog kataloga.
