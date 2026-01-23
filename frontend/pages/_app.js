import '../styles/globals.css';
import LeftSidebar from '../components/LeftSidebar';

export default function App({ Component, pageProps }) {
  return (
    <div className="appShell">
      <LeftSidebar />
      <main className="appMain">
        <Component {...pageProps} />
      </main>
    </div>
  );
}
