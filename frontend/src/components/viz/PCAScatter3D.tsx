"use client"

import { useMemo } from "react"
import { Html } from "@react-three/drei"

interface Point {
  x: number
  y: number
  z: number
  label?: string
  group?: number
  [key: string]: unknown
}

interface PCAScatter3DProps {
  points: Point[]
  pointSize?: number
  showLabels?: boolean
  colors?: string[]
}

/**
 * PCAScatter3D — 3D scatter plot for PCA visualization.
 * Points glow on hover and show details.
 */
export function PCAScatter3D({
  points,
  pointSize = 0.15,
  showLabels = false,
  colors = ["#FF6B35", "#00DCB4", "#9D4EDD", "#F59E0B", "#EF4444"],
}: PCAScatter3DProps) {
  // Normalize points to fit in view
  const normalizedPoints = useMemo(() => {
    if (points.length === 0) return []

    const xs = points.map((p) => p.x)
    const ys = points.map((p) => p.y)
    const zs = points.map((p) => p.z)

    const xRange = Math.max(...xs) - Math.min(...xs) || 1
    const yRange = Math.max(...ys) - Math.min(...ys) || 1
    const zRange = Math.max(...zs) - Math.min(...zs) || 1
    const maxRange = Math.max(xRange, yRange, zRange)

    return points.map((p) => ({
      ...p,
      x: (p.x - Math.min(...xs)) / maxRange - 0.5,
      y: (p.y - Math.min(...ys)) / maxRange - 0.5,
      z: (p.z - Math.min(...zs)) / maxRange - 0.5,
    }))
  }, [points])

  return (
    <group>
      {/* Axes */}
      <axesHelper args={[1.5]} />

      {/* Points */}
      {normalizedPoints.map((point, i) => {
        const color = point.group !== undefined ? colors[point.group % colors.length] : "#FF6B35"
        
        return (
          <group key={i} position={[point.x * 3, point.y * 3, point.z * 3]}>
            {/* Point sphere */}
            <mesh>
              <sphereGeometry args={[pointSize, 16, 16]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={0.5}
                transparent
                opacity={0.8}
              />
            </mesh>

            {/* Glow sprite */}
            <sprite>
              <spriteMaterial
                color={color}
                transparent
                opacity={0.3}
              />
            </sprite>

            {/* Label */}
            {showLabels && point.label && (
              <Html position={[0.2, 0.2, 0]} distanceFactor={10}>
                <div className="rounded bg-surface/90 px-2 py-1 text-xs text-text-primary backdrop-blur">
                  {point.label}
                </div>
              </Html>
            )}
          </group>
        )
      })}

      {/* Axis labels */}
      <Html position={[1.6, 0, 0]} distanceFactor={10}>
        <div className="text-xs text-text-muted">PC1</div>
      </Html>
      <Html position={[0, 1.6, 0]} distanceFactor={10}>
        <div className="text-xs text-text-muted">PC2</div>
      </Html>
      <Html position={[0, 0, 1.6]} distanceFactor={10}>
        <div className="text-xs text-text-muted">PC3</div>
      </Html>
    </group>
  )
}
