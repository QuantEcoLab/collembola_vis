# Prošireni ručni pregled v2

Novi paket: `outputs/ruler_exp02_os5_manual_validation_v2/`.
Web: **http://127.0.0.1:8771/**. Izvorni v1 pregled je na portu 8770.

## Sigurnosna kopija i granica prijenosa

Najnoviji backup za ovu migraciju:

`outputs/ruler_review_v1_backup_before_v2_20261103T153345321890Z/`

Snapshot sadrži **256 izvornih događaja i 56 spremljenih odluka**:
31 `ACCEPTED_UNCHANGED`, 6 `MANUALLY_CORRECTED`, 0 `FALSE_DETECTION`,
19 `AMBIGUOUS_OR_TRUNCATED`, 0 `UNLABELED_NEIGHBOUR`.

Tijekom pripreme spremljena je dodatna odluka za detection 185. Zato se koristi
ovaj noviji backup s 56 odluka, koji uključuje svih prvotnih 55.

V2 dobiva vlastitu bazu. Izvorni događaji, njihovi ID-jevi, payloadovi, timestampovi
i hash chain kopiraju se doslovno, zajedno sa svim povijesnim JSON artefaktima.
Izvorni fingerprint ostaje zabilježen u metapodacima podrijetla. Novi fingerprint
odnosi se na v2 cropove i predikcije. Novi `MIGRATE_CROP` događaji bilježe promjenu
koordinatnog sustava i vezu na izvornu odluku. Stara v1 baza se ne migrira.

## Proširenje i inferenca

- Isti 779 source/detection parovi i izvorni redoslijed.
- Iste source-level uloge: 567 train, 85 val, 127 test.
- Padding **220 originalnih piksela sa svake strane**, ograničen samo rubom
  fotografije. To je veličina ranije korištena u 12-case referentnom paketu.
- Cropovi se izrezuju iz originalnih full-resolution fotografija i spremaju kao
  PNG bez resizea. Jednakost stare regije sa starim cropom provjerena je za svih
  779 slučajeva.
- Oba segmentacijska modela ponovno se izvršavaju na proširenim slikama.
  `imgsz_policy` određuje stvarnu veličinu ulaza:
  `max(640, ceil(max(width,height)/32)*32)`, tako da letterbox ne smanjuje izvorne
  detalje proširenog cropa. Naslijeđeno v1 polje `imgsz` nije korišteno za ovu
  inferencu; svaki prediction JSON bilježi stvarni `imgsz`.
- Confidence prag 0.15; CPU; Ultralytics 8.4.175.
- Početni V7 SHA-256:
  `a46e37b85efe2794d1c5d72cabbbb6ef0aab5d5c72cd22d9264faeea46098883`.
- Fiksirani exp02_os5 SHA-256:
  `9540b6478a8d77fcf6f2aa7db5c639c812e3c91953baa57c58670903221fc958`.

Za oba modela završeno je 779/779 poziva. Početni V7 daje odabranu ciljnu masku
za 533 cropa, exp02_os5 za 552. Ostali imaju eksplicitno praznu ciljnu masku;
ne zamjenjuju se starim predikcijama ili maskama susjednih jedinki. To nisu
automatske odluke `FALSE_DETECTION` — slučaj i dalje pregledava korisnica.

Proširenje može otkriti druge jedinke. Za oba modela odabire se instanca čija maska
ima najveći IoU s izvornim detection pravokutnikom; confidence razrješava
izjednačenja. Bez preklapanja nema odabrane ciljne maske. Svi kandidati i njihovi
scoreovi zapisani su u prediction JSON-u. Rasterizacija i izdvajanje konture koriste
zamrznute legacy helpere. Ovo nije novo pokretanje detektora ni deduplikacije.

## Pravila prijenosa

Za svaku točku koristi se isključivo:

```text
x_v2 = x_v1 + x_origin_v1 - x_origin_v2
y_v2 = y_v1 + y_origin_v1 - y_origin_v2
```

Nema clippinga, zaokruživanja ili zaglađivanja oznaka. Provjerava se povratni
pomak s tolerancijom 1e-9 px za floating-point aritmetiku. Izvorne koordinate su
uvijek sačuvane u doslovno kopiranim povijesnim revizijama.

- **37 pozitivnih odluka** prenosi se s istom kategorijom, bilješkom i potvrdom;
  kontura se samo pomiče. Novi prošireni kontekst treba potvrditi prije buduće
  upotrebe u treningu (`expanded_context_review_required`).
- `FALSE_DETECTION` ostaje završena odluka za isti cilj.
- Nejasni/odrezani slučajevi vraćaju se na `PENDING` samo uz izričit dokaz:
  bilješku o odrezivanju/premalom cropu, prethodnu ljudsku `TRUNCATED` dijagnozu
  ili izričitu korisničku prijavu.
- Za ovu migraciju to su `C_1_2_Collembola005` detections **43**, **65**
  (bilješka „premali crop”) i **185** (korisnička prijava odrezane glave).
- Ostalih **16** nejasnih odluka bez takvog dokaza ostaje završeno u istoj
  kategoriji; razlog se ne pretpostavlja.
- `UNLABELED_NEIGHBOUR` zahtijeva ponovni pregled proširenog konteksta i postaje
  `PENDING`; u ovom snapshotu ih nema.
- Eventualni pending nacrti također se prenose čistim pomakom.

Rezultat inicijalnog prijenosa: **53 završene prenesene odluke, 3 vraćene na
PENDING**, svih 56 izvornih odluka i svih 256 događaja sačuvano u povijesti.

### Dopuna naknadno spremljene odluke 188

Nakon početnog snapshota u v1 je spremljena još odluka za detection **188**:
`AMBIGUOUS_OR_TRUNCATED`, bez bilješke o uzroku. Napravljen je završni backup:

`outputs/ruler_review_v1_backup_before_v2_20261103T154352095117Z/`

On sadrži **57 odluka / 257 događaja** (31 prihvaćena, 6 ispravljenih, 20 nejasnih).
Naknadna odluka prenesena je kao zasebna append-only dopuna, bez prepisivanja
početne v2 baze ili rebasea njezina hash chaina. Cijeli novi v1 snapshot i
neizmijenjeni izvorni JSON artefakti sačuvani su pod `parent_supplements/`.
Novi `MIGRATE_CROP` događaj sadrži i doslovan izvorni redak odluke 188, uključujući
njezin izvorni ID, timestamp, payload i hashove. Izvorne revizije prikazuju se
i u API povijesti slučaja. Dopuna odbija nadjačati slučaj koji već ima v2 izmjene.

**Konačno stanje nakon dopune: 54 prenesene završene odluke i 3 vraćene na
PENDING.** Dovršene kategorije u v2: 31 prihvaćena, 6 ispravljenih i 17 nejasnih.
Odluka 188 nije automatski vraćena na `PENDING`, jer uzrok odrezivanja za nju
nije zabilježen.

## Pregled i revizije

Brzi pregled i tipke **1–5**, **B/N**, **←/→**, **U** ostaju dostupni.
U za prenesenu odluku vraća nacrt u v2 koordinatama, čuva staru v1 odluku i novi
migracijski događaj. V2 je jasno označen u zaglavlju, a na originalnom prikazu
cijan isprekidani okvir označava granice starog cropa.

VAL/TEST i dalje skrivaju modele prije zaključavanja ljudske oznake. Ranija
izloženost modelima nasljeđuje se; promjena cropa ne vraća slijepi status.

## Izvještaji i naredbe

- `migration_report.json`: odluka za svaki preneseni slučaj, izvorna revizija,
  offset i dokaz za vraćanje na `PENDING`.
- `logs/verification.json`: identičnost stare povijesti, broj slučajeva i
  predikcija, uloge i aktualne kategorije.
- `prediction_seal.json`, `frozen_inputs/bundle_seal.json`: hashovi ulaza i oba
  nova skupa predikcija.
- `parent_snapshot/`: potpuna kopija ulaznog backupa ove migracije.
- `parent_supplements/` i `supplement_report.json`: kasniji snapshot i dokaz
  append-only prijenosa dodatne odluke 188.

```bash
# Read-only provjere izvora i predikcija; zapisuje samo v2 verification log.
python3 scripts/ruler_v7/prepare_expanded_controlled_review.py --stage verify

python3 scripts/ruler_v7/web_controlled_ruler_review.py \
  --output outputs/ruler_exp02_os5_manual_validation_v2 --port 8771

python3 -m unittest discover -s scripts/ruler_v7 -p test_expanded_controlled_review.py -v
python3 -m unittest discover -s scripts/ruler_v7 -p test_web_controlled_ruler_review.py -v
```

Alat za pripremu odbija prepisati postojeći output, završenu inferencu ili
postojeću v2 bazu. Modeli se učitavaju samo za inferencu. Nije izrađen trening
dataset i nije pokrenut trening.
