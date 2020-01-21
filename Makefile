.PHONY: sync
sync: get-files put-files

.PHONY: sync-remote
put-files:
	rsync -avzzru $$(pwd)/ datalab.imand3r.io:w207/
	
.PHONY: sync-local
get-files:
	rsync -avzzru datalab.imand3r.io:w207/ $$(pwd)
