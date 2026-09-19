import { Calculator } from 'lucide-react'

import { Button } from '@/components/ui/button'

export default function App() {
  return (
    <div className="site-shell">
      <main className="site-main">
        <Button className="mt-10">
          <Calculator />
          Scaffold check
        </Button>
      </main>
    </div>
  )
}
