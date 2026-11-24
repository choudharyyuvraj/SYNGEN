import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import { getHealth } from './api'

const HealthContext = createContext(null)

export function HealthProvider({ children }) {
  const [health, setHealth] = useState(null) // null = first check pending
  const [offline, setOffline] = useState(false)
  const timer = useRef(null)

  const refresh = useCallback(async () => {
    clearTimeout(timer.current)
    let next = 15000
    try {
      const h = await getHealth()
      setHealth(h)
      setOffline(false)
      if (h.mode === 'loading') next = 2000
    } catch {
      setOffline(true)
      next = 4000
    }
    timer.current = setTimeout(refresh, next)
  }, [])

  useEffect(() => {
    refresh()
    return () => clearTimeout(timer.current)
  }, [refresh])

  return <HealthContext.Provider value={{ health, offline, refresh }}>{children}</HealthContext.Provider>
}

export const useHealth = () => useContext(HealthContext)
