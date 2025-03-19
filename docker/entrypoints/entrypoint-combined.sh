#!/bin/sh
nohup llamafactory-cli webui > output.out 2> error.err 2>&1 &
nohup llamafactory-cli api > output.out 2> error.err 2>&1 &
nginx -c /etc/nginx/sites-available/nginx.conf
tail -f output.out