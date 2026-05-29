import { useState } from 'react';
import config from './portfolio.config.json';
import Profile from './features/profile/Profile';
import Rag from './features/rag/Rag';
import './global.css';
import './features/profile/Profile.module.css';

type ModuleId = 'profile' | string;

export default function App() {
  const [active, setActive] = useState<ModuleId>('profile');

  const activeModule = config.modules.find(m => m.id === active);

  function renderView() {
    if (active === 'profile') {
      return (
        <Profile
          profile={config.profile}
          modules={config.modules as any}
          onNavigate={(id) => setActive(id)}
        />
      );
    }
    if (active === 'rag') {
      return (
        <Rag
          description={activeModule?.description}
          stack={activeModule?.stack as any}
        />
      );
    }
    // Future modules go here
    return null;
  }

  return (
    <div className="portfolio-shell">

      {/* ── Top nav ── */}
      <nav className="portfolio-nav">
        <span className="portfolio-nav__wordmark">{config.profile.name}</span>
        <span className="portfolio-nav__divider" />

        {/* Profile tab */}
        <button
          className={`portfolio-nav__tab ${active === 'profile' ? 'portfolio-nav__tab--active' : ''}`}
          onClick={() => setActive('profile')}
        >
          <span className="portfolio-nav__pip" />
          Profile
        </button>

        {/* Module tabs */}
        {config.modules.map(m => {
          const isLive     = m.status === 'live';
          const isActive   = active === m.id;
          return (
            <button
              key={m.id}
              className={[
                'portfolio-nav__tab',
                isActive   ? 'portfolio-nav__tab--active'  : '',
                !isLive    ? 'portfolio-nav__tab--disabled' : '',
              ].join(' ')}
              onClick={() => isLive && setActive(m.id)}
              disabled={!isLive}
            >
              <span className="portfolio-nav__pip" />
              {m.label}
              {!isLive && <span className="portfolio-nav__badge">soon</span>}
            </button>
          );
        })}
      </nav>

      {/* ── Active view ── */}
      <div className="portfolio-view">
        {renderView()}
      </div>

    </div>
  );
}