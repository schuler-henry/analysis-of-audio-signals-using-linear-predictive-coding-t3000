import { useEffect, useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf'
import Image from 'next/image'
import styles from '../styles/Home.module.css'
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import Head from 'next/head';


export default function Home() {
  const [numPages, setNumPages] = useState<number>(0);
  const [clientWidth, setClientWidth] = useState(0);
  const [pdfViewerEnabled, setPdfViewerEnabled] = useState<boolean|undefined>(undefined);

  // set initial clientWidth and add resize listener
  useEffect(() => {
    setClientWidth(window.innerWidth);
    window.addEventListener('resize', () => {
      setClientWidth(window.innerWidth);
    });
    setPdfViewerEnabled(navigator.pdfViewerEnabled);
  }, [])

  function onDocumentLoadSuccess({ numPages } : { numPages: number }) {
    setNumPages(numPages);
  }
  return (
    <>
      <Head>
        <title>T3000 - Analyse von Audiosignalen unter der Verwendung von Linear Predictive Coding</title>
        <meta name="description" content="Projektarbeit T3000 von Henry Schuler" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main>
        {
        pdfViewerEnabled ?
          <div style={{ position: "absolute", left: "0", right: "0", bottom: "0", top: "0", overflow: "hidden" }}>
            <iframe src="./main.pdf" width="100%" height="100%" frameBorder={0}>
            </iframe>
          </div>
          :
          <div className={styles.wrapper}>
            <Document 
              className={styles.document}
              file="main.pdf" 
              onLoadSuccess={onDocumentLoadSuccess}
              options={{
                cMapUrl: `https://unpkg.com/pdfjs-dist@${pdfjs.version}/cmaps/`,
                cMapPacked: true,
                standardFontDataUrl: `https://unpkg.com/pdfjs-dist@${pdfjs.version}/standard_fonts`,
              }}
            >
              {[...Array(numPages)].map((_, i) => (
                <Page 
                  width={clientWidth}
                  key={"pdf-page-" + i}
                  pageNumber={i + 1} 
                  className={styles.page}
                />
              ))}
            </Document>
              <button 
                className={styles.downloadButton}
                onClick={() => {
                  // download main.pdf
                  window.open("./main.pdf", "_blank");
                }}
                >
                <Image
                  src="/download-install-line-icon.svg"
                  alt="Download Logo"
                  className={styles.vercelLogo}
                  width={20}
                  height={20}
                  priority
                />
              </button>
          </div>
        }
      </main>
    </>
  )
}
