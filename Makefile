.PHONY: dev backend frontend install

# Run both backend and frontend dev servers
dev:
	@echo "Starting backend and frontend..."
	$(MAKE) backend &
	$(MAKE) frontend &
	wait

backend:
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 9000

frontend:
	cd frontend && npm run dev

install:
	pip install -r backend/requirements.txt
	cd frontend && npm install
