.PHONY: install test synthetic clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

# Full pipeline on the synthetic fixture: no download needed.
# Writes to synthetic/, never to out/ (out/ holds real results).
synthetic:
	python scripts/03_slip.py --synthetic --out synthetic/
	python scripts/04_megathrust_strain.py --synthetic --out synthetic/ \
	    --spacing 0.12 --offset 10 --stride 5

clean:
	rm -rf synthetic/ .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
