import React from 'react';
import Link from 'next/link';

export default function Custom404() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-b from-muse-cream to-white">
      <h1 className="text-6xl font-serif font-bold text-muse-dark mb-4">404</h1>
      <p className="text-2xl text-gray-700 font-serif mb-8">Page not found ✨</p>
      <Link href="/">
        <button className="px-8 py-3 bg-muse-rose text-white rounded-full font-serif font-bold hover:shadow-lg transition-shadow">
          Return to Home
        </button>
      </Link>
    </div>
  );
}
