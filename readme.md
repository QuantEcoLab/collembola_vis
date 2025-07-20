


- Razvoj DL metode za detekciju i segmentaciju skokuna
- https://github.com/QuantEcoLab/collembolae_vis

Imamo tri slike sa skokunima, - zadatak je izraditi program za segmentaciju skokuna na slikama.

1. Jana je mukotrpno napravila set podataka s oznakama skokuna!
2. Algoritam za detekciju suspektnih mjesta gdje bi skokuni mogli biti
	1. prva moguća metoda -> blob detection algoritam <- Jana implementira trenutno
	2. druga moguća metoda -> segmetnacija watershed-algoritmom
3. Treba složiti skript koji će pripremiti dataset za trening, testiranje i validaciju

---> Tu smo
1. Istražiti opcije za izradu mjerenja veličine jedinke <- automatska segmentacija skokuna na odsječku jedne jedinke

2. Konvolucijska neuronska mreža <- detekcija ima li odsječak skokuna na sebi ili ne
3. Segmentacija 
	1. CV ()
	2. U-net
	3. Grounded-SAM 2
	4. VLLM

Zadatak: 20. srpanj
U pythonu funkcija za izračun volumena valjka (širina i dužina).