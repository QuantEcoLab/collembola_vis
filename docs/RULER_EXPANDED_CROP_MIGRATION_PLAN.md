# Plan prijenosa odluka na proširene cropove — bez izvršene migracije

## Zamrznuto stanje prije nadogradnje brzog pregleda

Read-only snapshot baze sadrži **55 dovršenih odluka**, 55 `DECISION` događaja
i ukupno 255 povijesnih događaja (zadnji ID: 255).

| Kategorija | Broj | Plan za eventualni prošireni crop |
| --- | ---: | --- |
| ACCEPTED_UNCHANGED | 31 | Kandidati za geometrijski prijenos konture pomakom |
| MANUALLY_CORRECTED | 6 | Kandidati za geometrijski prijenos ljudske konture pomakom |
| FALSE_DETECTION | 0 | Odluka za isti ciljni detection može ostati |
| AMBIGUOUS_OR_TRUNCATED | 18 | Odvojeni popis za trijažu / ponovni pregled |
| UNLABELED_NEIGHBOUR | 0 | Obvezna nova provjera šireg konteksta prije prijenosa |

Backup direktorij, relativno korijenu repozitorija:

`outputs/ruler_exp02_os5_manual_validation_v1/backup_before_fast_review_20261103T151444365814Z/`

Sadrži konzistentan SQLite backup (`review.sqlite3`), svih 255 pripadajućih
`annotation_revisions/*.json`, kopije `cases.json` i `config.json`, te
`backup_report.json`. SQLite `integrity_check`: `ok`.

SHA-256 baze u backupu:
`369e002fa505379c9d2324fd44ed7471747a5d16a9e195b8307984cdf6ad2a91`

SHA-256 kanonskog popisa svih izvornih redaka događaja:
`ed3e3afff8f668e084d60c471184678ad7e610de27013f4328730d81ef01b589`

## Uvjeti geometrijskog prijenosa

Za izvorni crop s početkom `(x_old, y_old)` i novi crop s početkom
`(x_new, y_new)` u istoj izvornoj fotografiji:

```text
x_vertex_new = x_vertex_old + x_old - x_new
y_vertex_new = y_vertex_old + y_old - y_new
```

Prijenos je dopušten samo kada novi crop u cijelosti sadrži stari, koristi istu
fotografiju, istu orijentaciju i istu pikselnu skalu. Treba provjeriti jednakost
piksela stare regije unutar nove slike i očuvanje površine/oblika nakon pomaka.
Ne primjenjivati resize, clipping, zaglađivanje ili interpolaciju konture.

Ovih **37 pozitivnih oznaka** jesu kandidati za prijenos geometrije. Širi crop
može otkriti nove neoznačene susjede ili dio tijela koji je prije bio odrezan;
zbog toga se podobnost novog cropa za trening ne smije naslijediti samo na
temelju pomaknute konture. Predvidjeti provjeru novog konteksta i odvojeno
bilježenje potvrda za novi crop.

`FALSE_DETECTION` odnosi se na isti ciljni source/detection par. Eventualno
očuvanje te odluke ne znači da je cijeli prošireni crop prazna pozadina.

## Odvojeni popis za ponovni pregled

Svi donji slučajevi imaju izvorni source ID **`C_1_2_Collembola005`** i
kategoriju `AMBIGUOUS_OR_TRUNCATED`. Ovo je dokumentirani radni popis;
trenutačna baza i njezine odluke nisu vraćene na `PENDING`.

| Detection ID | Izvorna revizija odluke | Izvorna bilješka | Planirana trijaža |
| --- | ---: | --- | --- |
| 15 | 70 | prazno | Provjeriti razlog nejasnoće / odrezanost |
| 32 | 120 | prazno | Provjeriti razlog nejasnoće / odrezanost |
| 37 | 121 | prazno | Provjeriti razlog nejasnoće / odrezanost |
| 43 | 123 | premali crop | Ponovni pregled na proširenom cropu |
| 65 | 136 | premali crop | Ponovni pregled na proširenom cropu |
| 69 | 152 | prazno | Provjeriti razlog nejasnoće / odrezanost |
| 76 | 171 | prazno | Provjeriti razlog nejasnoće / odrezanost |
| 116 | 207 | prazno | Provjeriti razlog nejasnoće / odrezanost |
| 120 | 211 | prazno | Provjeriti razlog nejasnoće / odrezanost |
| 134 | 221 | prazno | Provjeriti razlog nejasnoće / odrezanost |
| 147 | 234 | prazno | Provjeriti razlog nejasnoće / odrezanost |
| 150 | 235 | prazno | Provjeriti razlog nejasnoće / odrezanost |
| 151 | 238 | prazno | Provjeriti razlog nejasnoće / odrezanost |
| 155 | 239 | prazno | Provjeriti razlog nejasnoće / odrezanost |
| 157 | 243 | prazno | Provjeriti razlog nejasnoće / odrezanost |
| 165 | 247 | prazno | Provjeriti razlog nejasnoće / odrezanost |
| 167 | 248 | prazno | Provjeriti razlog nejasnoće / odrezanost |
| 173 | 252 | prazno | Provjeriti razlog nejasnoće / odrezanost |

Dva slučaja imaju izričitu bilješku o premalom cropu. Za ostalih 16 razlog
nije zapisan; kategorija sama ne dokazuje da je uzrok odrezani rub. U budućem
novom paketu sve dokazano odrezane slučajeve izdvojiti kao `PENDING` za ponovni
pregled, uz poveznicu na staru odluku. Ostale trijažirati, bez izmišljanja razloga.

`UNLABELED_NEIGHBOUR` nikada ne prenositi kao završenu odluku bez nove provjere
šireg konteksta, čak ni kada se ciljni poligon može točno pomaknuti.

## Predloženi budući postupak, uz zasebno odobrenje

1. Ponovno napraviti backup i ispisati tada aktualne kategorije i konkretan
   popis ulaznih/izlaznih datoteka prije migracije.
2. Definirati granice proširenja i provjeriti source-level train/val/test uloge.
3. Pripremiti odvojeni pregledni paket; staru bazu, slike i povijest zadržati kao
   izvorni dokaz. Očuvati source ID i detection ID. Verziju cropa voditi odvojeno.
4. Uz svaku prenesenu oznaku zapisati stari event/label hash, oba crop origin-a,
   dimenzije, hashove slike i pomak. Izvorni timestamp ostaje provenancija;
   novi događaj prijenosa ima svoj timestamp.
5. Geometrijski prenijeti pozitivne konture samo uz navedene provjere; negativne
   i nejasne kategorije obraditi prema tablici. Ne izvoziti dataset iz plana.
6. Napraviti izvještaj prije/poslije: kategorije, iznimke, lista za ponovni
   pregled i dokaz da su svi izvorni događaji ostali identični.

VAL/TEST ostaju izvan treninga. Ako su predikcije već bile otkrivene na staroj
verziji slučaja, tu izloženost treba naslijediti u novom paketu; proširenje cropa
ne vraća prethodno model-exposed slučaj u prvobitno slijepi status.

## Datoteke nadogradnje brzog pregleda

- `scripts/ruler_v7/web_controlled_ruler_review.html`
- `scripts/ruler_v7/web_controlled_ruler_review.py`
- `scripts/ruler_v7/test_web_controlled_ruler_review.py`
- `docs/CONTROLLED_RULER_REVIEW.md`
- `docs/RULER_EXPANDED_CROP_MIGRATION_PLAN.md` (ovaj novi dokument)

Nadogradnja uvodi nove događaje `REOPEN` tek kada korisnica pritisne U.
Postojeći redci i njihove JSON revizije ostaju izvorni. Schema baze ne treba
migraciju za novi tip događaja.
