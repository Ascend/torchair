---
name: default-skills
description: |
  **必须触发此 default-skills 的场景**（用户提到以下任何内容时使用）：找不到/缺失/安装/更新 gitcode-pr, gitcode-issue, api-doc-generator, gitcode-pipeline skill时，或者安装必备/默认/必要/必须skills时；
---

# 安装默认skills步骤

1. 读取`.claude/skills/default-skills/scripts/install-default-skills.sh` 获取 `DEFAULT_SKILLS`
2. 首先尝试使用 `.claude/skills/default-skills/scripts/install-default-skills.sh` 安装或更新skills，执行后检查 `.claude/skills/_remote/`目录有没有`DEFAULT_SKILLS`，如果有立即结束；如果没有，继续下一步
3. 使用git克隆`https://gitcode.com/cann-agent/skills.git` 到临时目录，要使用`--depth 1`参数，该仓下找到`DEFAULT_SKILLS`，拷贝到`.claude/skills/_remote/`目录下

## 默认skills使用场景

1. **必须触发 gitcode-issue 的场景**（用户提到以下任何内容时使用）：

   - 查看/读取 issue：查看issue、看看issue、读取issue、打开issue、issue详情、issue是什么
   - GitCode URL：gitcode.com/**/issues/**、cann/ge/issues、issue链接
   - 直接说编号：issue 123、#123、问题123
   - 查看评论：issue评论、评论内容

2. **必须触发此 gitcode-pr 的场景**（用户提到以下任何内容时使用）：

   - 创建/提交 PR：创建PR、提个PR、发PR、做个PR、帮我PR、生成PR、需要PR、pull request、merge request
   - 推送代码到远程：push代码、推代码、把代码推上去、提交到远程、推送到gitcode、提交代码到GitCode
   - 合并请求：合并请求、代码合入请求、请求合并、merge request
   - PR模板/描述：PR模板、PR描述、PR格式
   - 关联issue创建PR：关issue的PR、关联issue创建PR
   - 获取PR改动：查看PR变更、PR文件列表、PR改了什么、看PRdiff、获取PR文件
   - **获取 PR 评论**：查看PR评论、PR评论、获取评论、read comments
   - **查看 PR 讨论**：PR discussions、查看讨论、discussions
   - **删除 PR 评论**：删除评论、删除PR评论、移除评论、delete comment、移除这条评论

3. **必须触发此 api-doc-generator 的场景**（用户提到以下任何内容时使用）：
用于生成对外api文档，当用户说生成接口文档，生成接口说明，增加接口说明，增加接口文档时使用该skill

4. **必须触发此 gitcode-pipeline 的场景**（用户提到以下任何内容时使用）：

   - 触发流水线、启动CI、跑流水线、触发pipeline
   - 查看流水线状态、流水线结果、CI状态、pipeline状态
   - 等待流水线结果、盯流水线、盯CI、看一下PR的CI
   - 流水线失败、CI失败、pipeline失败
