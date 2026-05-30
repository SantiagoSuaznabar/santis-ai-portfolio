import { useState } from 'react';
import type { FC } from 'react';
import styles from './Profile.module.css';

// ─── Types ────────────────────────────────────────────────────────────────────

interface TechCategory {
  category: string;
  color: string;
  items: string[];
}

interface Language {
  name: string;
  level: string;
}

interface Certification {
  name: string;
  issuer: string;
  year?: string;
  url?: string;
}

interface ProfileData {
  name: string;
  title: string;
  bio: string;
  github?: string;
  linkedin?: string;
  email?: string;
  avatar?: string;
  techStack?: TechCategory[];
  languages?: Language[];
  certifications?: Certification[];
}

interface Module {
  id: string;
  label: string;
  status: 'live' | 'coming-soon';
  description?: string;
  stack?: { name: string; role: string; color?: string }[];
}

interface Props {
  profile: ProfileData;
  modules: Module[];
  onNavigate: (id: string) => void;
}

const CERTS_PER_PAGE = 4;

// ─── Component ────────────────────────────────────────────────────────────────

const Profile: FC<Props> = ({ profile, modules, onNavigate }) => {
  const [certPage, setCertPage] = useState(0);

  const liveModules = modules.filter(m => m.status === 'live');
  const soonModules = modules.filter(m => m.status === 'coming-soon');

  const hasTech  = (profile.techStack?.length   ?? 0) > 0;
  const hasLangs = (profile.languages?.length   ?? 0) > 0;
  const hasCerts = (profile.certifications?.length ?? 0) > 0;

  const totalCertPages = hasCerts
    ? Math.ceil(profile.certifications!.length / CERTS_PER_PAGE)
    : 0;
  const visibleCerts = hasCerts
    ? profile.certifications!.slice(certPage * CERTS_PER_PAGE, (certPage + 1) * CERTS_PER_PAGE)
    : [];

  return (
    <div className={styles.page}>
      <div className={styles.inner}>

        {/* ── Hero ── */}
        <section className={styles.hero}>
          <div className={styles.avatarWrap}>
            {profile.avatar
              ? <img src={profile.avatar} alt={profile.name} className={styles.avatarImg} />
              : <div className={styles.avatarFallback}>{profile.name.charAt(0)}</div>
            }
            <span className={styles.availableDot} title="Open to opportunities" />
          </div>

          <div className={styles.heroText}>
            <h1 className={styles.name}>{profile.name}</h1>
            <p className={styles.title}>{profile.title}</p>
          </div>

          <div className={styles.links}>
            {profile.github && (
              <a href={profile.github} target="_blank" rel="noreferrer" className={styles.link}>
                <GithubIcon /><span>GitHub</span>
              </a>
            )}
            {profile.linkedin && (
              <a href={profile.linkedin} target="_blank" rel="noreferrer" className={styles.link}>
                <LinkedinIcon /><span>LinkedIn</span>
              </a>
            )}
            {profile.email && (
              <a href={`mailto:${profile.email}`} className={styles.link}>
                <MailIcon /><span>{profile.email}</span>
              </a>
            )}
          </div>
        </section>

        {/* ── Row 1: Bio + Tech Stack ── */}
        <div className={styles.twoColRow}>
          <section className={styles.leftCol}>
            <h2 className={styles.sectionLabel}>About</h2>
            <p className={styles.bio}>{profile.bio}</p>
          </section>

          {hasTech && (
            <section className={styles.rightCol}>
              <h2 className={styles.sectionLabel}>Tech Stack</h2>
              <div className={styles.techGrid}>
                {profile.techStack!.map(cat => (
                  <div key={cat.category} className={styles.techRow}>
                    <span
                      className={styles.techCategoryLabel}
                      style={{ '--cat-color': cat.color } as React.CSSProperties}
                    >
                      {cat.category}
                    </span>
                    <div className={styles.techItems}>
                      {cat.items.map(item => (
                        <span
                          key={item}
                          className={styles.techPill}
                          style={{ '--pill-color': cat.color } as React.CSSProperties}
                        >
                          {item}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>

        {/* ── Row 2: Languages + Certifications ── */}
        {(hasLangs || hasCerts) && (
          <div className={styles.twoColRow}>

            {hasLangs && (
              <section className={styles.leftCol}>
                <h2 className={styles.sectionLabel}>Languages</h2>
                <div className={styles.langsList}>
                  {profile.languages!.map(lang => (
                    <div key={lang.name} className={styles.langItem}>
                      <span className={styles.langName}>{lang.name}</span>
                      <span className={styles.langLevel}>{lang.level}</span>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {hasCerts && (
              <section className={styles.rightCol}>
                {/* Header row: label + pagination controls */}
                <div className={styles.sectionHeaderRow}>
                  <h2 className={styles.sectionLabel}>Certifications</h2>
                  {totalCertPages > 1 && (
                    <div className={styles.certPagination}>
                      <button
                        className={styles.certPageBtn}
                        onClick={() => setCertPage(p => p - 1)}
                        disabled={certPage === 0}
                        aria-label="Previous page"
                      >
                        ←
                      </button>
                      <span className={styles.certPageInfo}>
                        {certPage + 1} / {totalCertPages}
                      </span>
                      <button
                        className={styles.certPageBtn}
                        onClick={() => setCertPage(p => p + 1)}
                        disabled={certPage === totalCertPages - 1}
                        aria-label="Next page"
                      >
                        →
                      </button>
                    </div>
                  )}
                </div>

                {/* 2×2 cert grid */}
                <div className={styles.certPageGrid}>
                  {visibleCerts.map(cert =>
                    cert.url ? (
                      <a
                        key={cert.name + cert.url}
                        href={cert.url}
                        target="_blank"
                        rel="noreferrer"
                        className={styles.certCard}
                      >
                        <span className={styles.certName}>{cert.name}</span>
                        <span className={styles.certMeta}>
                          {cert.issuer}{cert.year ? ` · ${cert.year}` : ''}
                        </span>
                        <span className={styles.certLinkIcon}><ExternalLinkIcon /></span>
                      </a>
                    ) : (
                      <div key={cert.name} className={`${styles.certCard} ${styles.certCardNoLink}`}>
                        <span className={styles.certName}>{cert.name}</span>
                        <span className={styles.certMeta}>
                          {cert.issuer}{cert.year ? ` · ${cert.year}` : ''}
                        </span>
                      </div>
                    )
                  )}
                </div>
              </section>
            )}

          </div>
        )}

        {/* ── Live demos ── */}
        {liveModules.length > 0 && (
          <section className={styles.fullRow}>
            <h2 className={styles.sectionLabel}>Live demos</h2>
            <div className={styles.moduleGrid}>
              {liveModules.map(m => (
                <button key={m.id} className={styles.moduleCard} onClick={() => onNavigate(m.id)}>
                  <div className={styles.moduleCardTop}>
                    <span className={styles.moduleCardLabel}>{m.label}</span>
                    <span className={styles.liveChip}>live</span>
                  </div>
                  {m.description && <p className={styles.moduleCardDesc}>{m.description}</p>}
                  {m.stack && m.stack.length > 0 && (
                    <div className={styles.moduleCardStack}>
                      {m.stack.map(s => (
                        <span
                          key={s.name}
                          className={styles.stackPill}
                          style={{ '--pill-color': s.color || '#3b82f6' } as React.CSSProperties}
                        >
                          {s.name}
                        </span>
                      ))}
                    </div>
                  )}
                  <div className={styles.moduleCardCta}>Open demo <ArrowIcon /></div>
                </button>
              ))}
            </div>
          </section>
        )}

        {/* ── Coming soon ── */}
        {soonModules.length > 0 && (
          <section className={styles.fullRow}>
            <h2 className={styles.sectionLabel}>Coming soon</h2>
            <div className={styles.moduleGrid}>
              {soonModules.map(m => (
                <div key={m.id} className={`${styles.moduleCard} ${styles.moduleCardDim}`}>
                  <div className={styles.moduleCardTop}>
                    <span className={styles.moduleCardLabel}>{m.label}</span>
                    <span className={styles.soonChip}>soon</span>
                  </div>
                  {m.description && <p className={styles.moduleCardDesc}>{m.description}</p>}
                </div>
              ))}
            </div>
          </section>
        )}

      </div>
    </div>
  );
};

export default Profile;

// ── Icons ─────────────────────────────────────────────────────────────────────

function GithubIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.385-1.335-1.755-1.335-1.755-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 21.795 24 17.295 24 12c0-6.63-5.37-12-12-12z"/>
    </svg>
  );
}

function LinkedinIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
    </svg>
  );
}

function MailIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="4" width="20" height="16" rx="2"/>
      <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>
    </svg>
  );
}

function ArrowIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12h14M12 5l7 7-7 7"/>
    </svg>
  );
}

function ExternalLinkIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
      <polyline points="15 3 21 3 21 9"/>
      <line x1="10" y1="14" x2="21" y2="3"/>
    </svg>
  );
}