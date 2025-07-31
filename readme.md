


- Razvoj DL metode za detekciju i segmentaciju skokuna
- https://github.com/QuantEcoLab/collembolae_vis

Imamo tri slike sa skokunima, - zadatak je izraditi program za segmentaciju skokuna na slikama.

1. Jana je mukotrpno napravila set podataka s oznakama skokuna!
2. Algoritam za detekciju suspektnih mjesta gdje bi skokuni mogli biti
	1. prva moguća metoda -> blob detection algoritam <- Jana implementira trenutno
	2. druga moguća metoda -> segmetnacija watershed-algoritmom
3. Treba složiti skript koji će pripremiti dataset za trening, testiranje i validaciju


1. Istražiti opcije za izradu mjerenja veličine jedinke <- automatska segmentacija skokuna na odsječku jedne jedinke

2. Konvolucijska neuronska mreža <- detekcija ima li odsječak skokuna na sebi ili ne
3. Segmentacija 
	1. CV ()
	2. U-net
	3. Grounded-SAM 2
	4. VLLM



Zadatak: 20. srpanj
- [x] U pythonu funkcija za izračun volumena valjka (širina i dužina).

---> Tu smo

31.7.2025.

U branchu "sam". Treba napraviti novi folder "templates" u `data` folderu. folder treba sadržavati odsječke iz slike koju je Eugen označio. Slika je u data folderu. Svi označeni ROI (BBOX-ovi) su u priloženom.zip fajlu.

Nakon izrade dosječaka treba osmisliti/pronaći mjeru kojom se mogu međusobno razlikovati slike. Metoda za te mjere bi mogla biti nekakva "clustering" metoda. Probaj pronaći nekakavu metodu kou bi mogla to odraditi.

> napravili smo skoro 80% prirpreme sam da uspijemo napraviti program koji je koristan. A moglo bi i u publikaciju ići.