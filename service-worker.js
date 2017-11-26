var files = [
  'index.php',
  'lib.php',
  'admin.php',
  'login.php',
  'conn.php',
  'graph.php',
  'manifest.json',
  'files.json',
  'logout.php',
  'service-worker.js',
  'verifLogin.php',
  'view/css/estilo.css',
  'view/css/c3.min.css',
  'view/css/bootstrap.min.css',
  'view/css/font-awesome.min.css',
  'view/css/bootstrap.css',
  'view/css/tether.min.css',
  'view/fonts/fontawesome-webfont.eot',
  'view/fonts/fontawesome-webfont.svg',
  'view/fonts/fontawesome-webfont.ttf',
  'view/fonts/fontawesome-webfont.woff',
  'view/fonts/fontawesome-webfont.woff2',
  'view/fonts/FontAwesome.otf',
  'view/img/bg.png',
  'view/img/ico.png',
  'view/js/app.js',
  'view/js/bootstrap.js',
  'view/js/bootstrap.min.js',
  'view/js/c3.min.js',
  'view/js/d3.min.js',
  'view/js/graph.js',
  'view/js/install.js',
  'view/js/jquery.min.js',
  'view/js/spin.js',  
  'view/js/tether.min.js',
  'view/less/animated.less',
  'view/less/bordered-pulled.less',
  'view/less/core.less',
  'view/less/fixed-width.less',
  'view/less/font-awesome.less',
  'view/less/icons.less',
  'view/less/larger.less',
  'view/less/list.less',
  'view/less/mixins.less',
  'view/less/path.less',
  'view/less/rotated-flipped.less',
  'view/less/screen-reader.less',
  'view/less/stacked.less',
  'view/less/variables.less',
  'view/scss/_animated.scss',
  'view/scss/_bordered-pulled.scss',
  'view/scss/_core.scss',
  'view/scss/_fixed-width.scss',
  'view/scss/_icons.scss',
  'view/scss/_larger.scss',
  'view/scss/_list.scss',
  'view/scss/_mixins.scss',
  'view/scss/_path.scss',
  'view/scss/_rotated-flipped.scss',
  'view/scss/_screen-reader.scss',
  'view/scss/_stacked.scss',
  'view/scss/_variables.scss',
  'view/scss/font-awesome.scss'
];
// dev only
if (typeof files == 'undefined') {
  var files = [];
} else {
  files.push('./');
}

var CACHE_NAME = 'project_frozen1';

self.addEventListener('activate', function(event) {
  console.log('[SW] Activate');
  event.waitUntil(
    caches.keys().then(function(cacheNames) {
      return Promise.all(
        cacheNames.map(function(cacheName) {
          if (CACHE_NAME.indexOf(cacheName) == -1) {
            console.log('[SW] Delete cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

self.addEventListener('install', function(event){
  console.log('[SW] Install');
  event.waitUntil(
    caches.open(CACHE_NAME).then(function(cache) {
      return Promise.all(
      	files.map(function(file){
      		return cache.add(file);
      	})
      );
    })
  );
});

self.addEventListener('fetch', function(event) {
  console.log('[SW] fetch ' + event.request.url)
  event.respondWith(
    caches.match(event.request).then(function(response){
      return response || fetch(event.request.clone());
    })
  );
});

self.addEventListener('notificationclick', function(event) {
  console.log('On notification click: ', event);
  clients.openWindow('/');
});