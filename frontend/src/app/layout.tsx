import type { Metadata } from "next";
import { Geist_Mono, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DataLens AI",
  description: "Autonomous Data Analysis Platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <link
          href="https://api.fontshare.com/v2/css?f[]=satoshi@300,400,500,700,900&display=swap"
          rel="stylesheet"
        />
      </head>
      <body
        className={`${geistMono.variable} ${jetbrainsMono.variable} font-body antialiased`}
        style={{ fontFamily: 'var(--font-body), monospace' }}
      >
        <style jsx global>{`
          :root {
            --font-heading: 'Satoshi', sans-serif;
            --font-body: var(--font-geist-mono), monospace;
            --font-code: var(--font-jetbrains-mono), monospace;
          }
          h1, h2, h3, h4, h5, h6 {
            font-family: var(--font-heading), sans-serif !important;
          }
        `}</style>
        {children}
      </body>
    </html>
  );
}
