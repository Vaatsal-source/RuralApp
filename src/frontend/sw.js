self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open('caresathi-v1').then((cache) => {
      return cache.addAll(['index.html', 'triage.html', 'analysis.html', 'inventory.html']);
    })
  );
});

self.addEventListener('fetch', (e) => {
  e.respondWith(
    caches.match(e.request).then((res) => res || fetch(e.request))
  );
});