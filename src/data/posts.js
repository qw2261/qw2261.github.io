import { marked } from 'marked'

const rawModules = import.meta.glob('./posts/*.md', { query: '?raw', import: 'default', eager: true })

function parsePost(filePath, raw) {
  const id = filePath.replace('./posts/', '').replace('.md', '')
  const lines = raw.split('\n')

  let title = id
  let date = '2020-01-01'
  let contentStart = 0

  for (let i = 0; i < Math.min(lines.length, 10); i++) {
    const line = lines[i].trim()
    if (!title || title === id) {
      if (line.startsWith('## ')) {
        title = line.replace('## ', '').trim()
        contentStart = i + 1
        continue
      }
    }
    const dateMatch = line.match(/^(\d{4}[/-]\d{2}[/-]\d{2})/)
    if (dateMatch && contentStart > 0) {
      date = dateMatch[1].replace(/\//g, '-')
      contentStart = i + 1
      break
    }
  }

  const content = lines.slice(contentStart).join('\n').trim()

  return { id, title, date, content }
}

const allPosts = Object.entries(rawModules).map(([path, raw]) => parsePost(path, raw))

export const posts = allPosts
  .map(({ id, title, date }) => ({ id, title, date }))
  .sort((a, b) => new Date(b.date) - new Date(a.date))

const contentMap = new Map()
allPosts.forEach(p => contentMap.set(p.id, p.content))

export async function getPostById(id) {
  const meta = posts.find(p => p.id === id)
  if (!meta) return null

  const content = contentMap.get(id) || ''
  const htmlContent = marked(content)

  return {
    ...meta,
    content: htmlContent
  }
}
