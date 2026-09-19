import { Link } from 'react-router'

export default function NotFound() {
  return (
    <div className="flex flex-col gap-4 py-10">
      <h1 className="text-2xl font-medium tracking-tight">Page not found</h1>
      <Link to="/" className="text-sm underline underline-offset-4">
        Back to the tool list
      </Link>
    </div>
  )
}
