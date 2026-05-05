import { marked } from 'marked'

export const posts = [
  {
    id: 'noscience2',
    title: '没有科学的世界 (二）',
    date: '2026-05-03'
  },
  {
    id: 'qingmingsuibi',
    title: '清明随笔',
    date: '2020-04-04'
  },
  {
    id: 'younan-1',
    title: '囿南记事（篇一）',
    date: '2020-02-03'
  },
  {
    id: 'shanghai',
    title: '蝈蝈叫了一夏-回忆上海工作的日子',
    date: '2019-08-20'
  },
  {
    id: 'noscience',
    title: '没有科学的世界 (一）',
    date: '2018-07-11'
  }
]

const postModules = import.meta.glob('./posts/*.md', { query: '?raw', import: 'default' })

export async function getPostById(id) {
  const post = posts.find(p => p.id === id)
  if (!post) return null
  
  const filePath = `./posts/${id}.md`
  const content = await postModules[filePath]()
  const htmlContent = marked(content)
  
  return {
    ...post,
    content: htmlContent
  }
}