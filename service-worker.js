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
  'css/estilo.css',
  'css/c3.min.css',
  'css/bootstrap.min.css',
  'css/font-awesome.min.css',
  'css/bootstrap.css',
  'css/tether.min.css',
  'fonts/fontawesome-webfont.eot',
  'fonts/fontawesome-webfont.svg',
  'fonts/fontawesome-webfont.ttf',
  'fonts/fontawesome-webfont.woff',
  'fonts/fontawesome-webfont.woff2',
  'fonts/FontAwesome.otf',
  'img/bg.png',
  'img/ico.png',
  'js/app.js',
  'js/bootstrap.js',
  'js/bootstrap.min.js',
  'js/c3.min.js',
  'js/d3.min.js',
  'js/graph.js',
  'js/install.js',
  'js/jquery.min.js',
  'js/spin.js',  
  'js/tether.min.js',
  'less/animated.less',
  'less/bordered-pulled.less',
  'less/core.less',
  'less/fixed-width.less',
  'less/font-awesome.less',
  'less/icons.less',
  'less/larger.less',
  'less/list.less',
  'less/mixins.less',
  'less/path.less',
  'less/rotated-flipped.less',
  'less/screen-reader.less',
  'less/stacked.less',
  'less/variables.less',
  'scss/_animated.scss',
  'scss/_bordered-pulled.scss',
  'scss/_core.scss',
  'scss/_fixed-width.scss',
  'scss/_icons.scss',
  'scss/_larger.scss',
  'scss/_list.scss',
  'scss/_mixins.scss',
  'scss/_path.scss',
  'scss/_rotated-flipped.scss',
  'scss/_screen-reader.scss',
  'scss/_stacked.scss',
  'scss/_variables.scss',
  'scss/font-awesome.scss'
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