.PHONY: sync
sync: get-files put-files

.PHONY: sync-remote
put-files:
	rsync -avzzru $$(pwd)/ datalab.imand3r.io:w207/
	
.PHONY: sync-local
get-files:
	rsync -avzzru datalab.imand3r.io:w207/ $$(pwd)

.PHONY: jupyter
jupyter:
	docker start jupyter-tensorflow || \
		docker run -d -p 8888:8888 \
		-e JUPYTER_ENABLE_LAB=yes \
		-v $(PWD):/home/jovyan/work \
		--name jupyter-tensorflow \
		jupyter/tensorflow-notebook:7a0c7325e470
		@sleep 5
		@make --no-print-directory jupyter-url

.PHONY: jupyter-url
jupyter-url:
	$(eval URL := $(shell docker logs jupyter-tensorflow 2>&1 | egrep -o 'http://127.0.0.1:8888/\?token=[0-9a-f]+' | tail -1))
	@echo Jupyter URL: $(URL)

.PHONY: stop-jupyter
stop-jupyter:
	docker kill jupyter-tensorflow || true

.PHONY: kill-jupyter
kill-jupyter: stop-jupyter
	docker rm jupyter-tensorflow
