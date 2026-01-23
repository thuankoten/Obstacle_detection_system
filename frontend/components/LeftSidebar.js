import Link from 'next/link';
import { useRouter } from 'next/router';

function NavItem({ href, label }) {
  const router = useRouter();
  const active = router.pathname === href;
  return (
    <Link className={active ? 'navItem active' : 'navItem'} href={href}>
      {label}
    </Link>
  );
}

export default function LeftSidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">Obstacle Detection</div>
      <nav className="nav">
        <NavItem href="/detection/video" label="Detection Video" />
        <NavItem href="/detection/realtime" label="Detection Real Time" />
      </nav>
    </aside>
  );
}
