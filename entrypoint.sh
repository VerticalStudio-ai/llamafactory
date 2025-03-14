nohup llamafactory-cli api > output.out 2> error.err &
nginx -c /etc/nginx/sites-available/nginx.conf
tail -f output.out