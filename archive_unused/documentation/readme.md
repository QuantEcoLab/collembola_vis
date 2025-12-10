
# Collembola Detection Project (Croatian)

⚠️ **NOTE**: For full English documentation, see [README.md](README.md)

## Status: Implementirano ✓

Razvoj DL metode za detekciju i segmentaciju skokuna - **ZAVRŠENO**
- https://github.com/QuantEcoLab/collembolae_vis

### Implementirano:
- ✅ Automatska detekcija i segmentacija skokuna korištenjem SAM modela
- ✅ Template-guided pristup s NCC matching-om
- ✅ Automatsko mjerenje duljine, širine, površine i volumena
- ✅ Export rezultata u CSV i JSON formatu
- ✅ Vizualizacija s obojenim maskama
- ✅ Optimizacija performansi (auto-downscaling, subsampling)

### Glavni program: `sam_templates.py`

```bash
# Aktiviraj environment
conda activate collembola

# Pokreni detekciju
python sam_templates.py "data/slike/K1_Fe2O3001 (1).jpg" \
    --template-dir data/organism_templates \
    --sam-checkpoint checkpoints/sam_vit_b.pth \
    --auto-download \
    --output out/measurements.csv
```

Ili koristi pripremljeni primjer:
```bash
./run_example.sh
```

### Arhivirane skripte (stare verzije):
- `archive_old_scripts/mk_dataset.py` - Početna blob detection metoda
- `archive_old_scripts/measure_collembolas.py` - Watershed segmentacija
- `archive_old_scripts/sam_detect.py` - SAM s anotacijama
- `archive_old_scripts/sam_guided.py` - Prototype-based detekcija

### Napredak:
~~- [ ] Blob detection algoritam~~  
~~- [ ] Watershed segmentacija~~  
~~- [ ] Priprema dataseta za trening~~  
- ✅ **Template-guided SAM segmentacija** (FINALNA VERZIJA)
- ✅ **Automatska mjerenja volumena** (ellipsoid model)
- ✅ **Optimizacija performansi** (progress bars, auto-scaling)

### Sljedeći koraci:
- Validacija rezultata na svim slikama
- Priprema podataka za publikaciju
- Usporedba s ručnim mjerenjima

> Program je gotov i spreman za korištenje. Svi ciljevi su postignuti! 🎉