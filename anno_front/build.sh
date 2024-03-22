npm run fe-build
sudo rm /var/www/dist -rf
sudo mv dist /var/www
sudo service apache2 restart
