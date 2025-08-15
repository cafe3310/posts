> © 2025 [cafe3310](https://github.com/cafe3310) 本作品采用 [知识共享 署名-非商业性使用-相同方式共享 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hans) 进行许可。
> 
> © 2025 by [cafe3310](https://github.com/cafe3310). This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed). 

---

基于一些个人讨论，由 Gemini 2.5 Pro 研究生成

---

本报告是为调研金融场景下“大型语言模型（LLM）如何进行大规模数据分析，以及单智能体与多智能体系统架构优劣”这一项目可行性时的系统综述。

报告指出，LLM本身并不适合直接处理海量结构化数据，但可作为自然语言接口，自动生成SQL/Python等代码，由传统数据引擎完成精准计算；而RAG（检索增强生成）适合查找个体记录，无法胜任全局聚合。

面对更复杂的业务场景，单智能体开发和控制容易，但能力有上限；多智能体系统能实现复杂分析和协作，但也会带来不可控、混乱的风险，其团队管理难题正如真实组织中人员分工与沟通失调。

行业案例显示，成功的关键在于构建以LLM为接口、结合数据引擎、加强系统治理的复合型架构。

## 第一部分：LLM驱动的结构化数据分析基础方法论

本报告的第一部分旨在直接回应用户对于可靠性的核心关切。它首先解构了大型语言模型（LLM）在处理结构化数据时固有的挑战，然后详细阐述了两种弥合这一鸿沟的主要且最可靠的方法，为后续更复杂的架构讨论奠定基础。

### 1.1 可靠性评估：LLM与结构化数据之间的根本性错配

在探讨使用LLM进行支付成功率等关键业务指标分析之前，必须首先建立一个清醒的认知：直接将大型表格化金融数据“展示”给LLM并期望其进行精确分析，是一种极易失败的策略。这种方法的不可靠性源于LLM的核心设计与结构化数据分析任务之间的根本性错配。

LLM本质上是概率性的文本序列预测器，而非确定性的逻辑计算引擎。它们在公共基准测试中取得的成功，往往难以直接复制到真实世界的企业场景中。研究表明，企业环境引入了独特的挑战，如更大的数据规模、更复杂的任务逻辑以及对内部领域知识的依赖，而这些是现有基于LLM的方法常常忽视的 1。

实证研究为这一观点提供了有力支撑。微软的一项研究揭示了一个显著的性能鸿沟：即便是在相对简单的表格化数据任务上，如识别行列数量，像GPT-4这样的顶级模型所能达到的最高综合准确率也仅为65.43% 2。这一结果明确指出，LLM对表格结构只有基础的、远非完美的理解。当从处理公共网络数据转向处理企业特有的数据集（例如SAP的客户数据）时，性能下降得更为剧烈，F1分数可能骤降至0.07这样的个位数水平 1。这种性能的断崖式下跌揭示了问题的严重性。

导致这种失败的根本原因可以归结为以下几点：

● **分词器的低效性与歧义**：标准的分词器（Tokenizer），例如GPT-2所使用的，在处理结构化数据时会遇到严重问题。它们可能以不可预测的方式拆分数字（如“5,234”）和作为分隔符的标点符号（如CSV文件中的逗号），这使得模型难以对数值进行数学推理，也无法准确理解表格的列边界 3。

● **缺乏结构化意识**：LLM的“世界观”是基于文本流的，它们并不像数据库或电子表格软件那样，内在地理解“行”和“列”的二维结构概念。其核心的“理解”机制——向量嵌入（Vector Embeddings），虽然能捕捉语义相似性，但本身却缺失了表格数据所蕴含的严格结构信息 3。

● **企业特定上下文的缺失**：企业数据分析往往需要大量内部知识，例如自定义的语义类型、内部业务术语、产品代号等。这些知识在LLM的公共训练数据中是完全缺失的，导致模型无法真正理解数据列的业务含义，从而做出错误的解读或关联 1。

这些根本性缺陷在实践中表现为各种具体的失败模式。实验和个人经验都表明，即便是要求LLM执行生成CSV文件这样的简单任务，它也常常会犯下混淆行、计算错误条目、添加多余空行或违反指定格式等错误，后续的人工纠错过程既繁琐又低效 3。

这一切指向一个超越模型本身的重要结论：这个问题本质上是**架构性**的，而不仅仅是**规模性**的。研究清晰地表明，即便是目前最强大的模型（如GPT-4）也难以胜任 1。这意味着，单纯寄望于下一代更大、更强的模型（如GPT-5）来自动解决这一问题是不现实的。问题的根源不在于模型参数量的不足，而在于其核心架构与任务需求的不匹配。LLM被设计用于实现语言的流畅性，而非保证逻辑和数学的确定性。因此，解决方案不应是强迫LLM去“成为”一个数据库，而应是构建一个全新的

**系统**——在这个系统中，LLM扮演其最擅长的角色：一个强大的自然语言接口，而将结构化数据的处理与计算任务交还给为此专门设计的传统数据引擎。这一认知上的转变，将整个问题的焦点从“LLM能否分析数据？”转向了“我们如何构建一个系统，让LLM能够智能地驱动一个传统数据分析引擎？”，从而为后续所有架构设计奠定了基础。

### 1.2 代码生成策略：将LLM定位为自然语言到查询的翻译器

基于前述对LLM局限性的分析，业界发展出了一种最为稳健且被广泛采纳的模式，以执行对大规模结构化数据的**聚合分析**——这直接适用于用户提出的支付成功率分析场景。该模式的核心思想是“代码生成”（Code Generation）。

**核心概念**：该策略的精髓在于，我们不要求LLM直接计算分析结果，而是利用其强大的自然语言理解和代码生成能力，将用户的自然语言问题翻译成一段可执行的代码（例如Python Pandas脚本或SQL查询）。然后，这段代码在一个独立的、确定性的计算环境中（如数据库或数据处理集群）被执行，从而完成实际的分析任务 5。这种方法巧妙地绕开了LLM的上下文窗口限制和其固有的数学计算短板。

**工作流程**：一个典型的代码生成分析系统按以下步骤运作：

1. **接收自然语言查询**：系统接收用户的自然语言指令，例如：“请分析2023年第四季度，交易金额超过100美元的支付成功率，并按商户类别进行细分”。

2. **构建增强提示（****Prompt Augmentation****）**：为了让LLM能够生成准确的代码，系统需要向其提供必要的上下文。这不仅仅是用户的查询，更关键的是要包含数据库的**元数据（****Metadata****）**，如表结构（Schema）、列名、数据类型，甚至可以提供几行样本数据，以帮助LLM理解数据的具体格式和内容 6。

3. **代码生成**：LLM在这些元数据的指导下，将用户的自然语言查询“翻译”成一段精确的、可执行的代码。对于上述例子，它可能会生成一条包含SELECT, COUNT, CASE WHEN, WHERE, 和 GROUP BY子句的SQL查询语句。

4. **代码执行与结果返回**：系统将LLM生成的代码发送到真实的数据环境中执行（例如，一个SQL数据库）。数据库完成计算后，将最终的聚合结果（而不是海量的原始数据）返回给系统。

**优势分析**：

● **无限扩展性**：由于LLM本身从不接触完整的原始数据集，它只需要处理相对小得多的元数据信息。因此，这种方法的分析能力可以轻松扩展至PB级别的数据，因为真正的计算负载由为此专门设计的数据库或数据仓库承担 5。

● **计算的精确性**：该方法充分利用了传统数据库在数学计算上的确定性和高精度，完全避免了LLM在进行求和、平均、百分比计算时可能出现的“幻觉”（Hallucination）问题。

● **验证的可靠性**：这是目前业界公认的，在处理复杂分析查询上最为成熟和可靠的方法。实践证明，让LLM生成创建CSV文件的Python代码，其可靠性远高于直接要求LLM生成CSV内容本身 3。

**挑战与缓解措施**：

● **查询的复杂性**：尽管此方法非常强大，但当查询涉及到跨多个表的复杂连接（JOINs）或深层嵌套的逻辑时，LLM生成准确代码的能力会显著下降。这是当前研究领域的一个活跃且具有挑战性的方向，现有方法在多表连接场景下常常失败 5。缓解这一问题的方法包括：提供清晰、详尽的表结构和关系元数据，或采用更先进的多智能体系统（将在第二部分讨论）来将复杂查询分解为多个简单的子查询。

● **严重的安全风险**：自动执行由LLM生成的代码存在巨大的安全隐患。如果不对生成的代码进行审查，恶意代码（如删除数据库、窃取数据）可能会对系统造成毁灭性打击 6。缓解这一风险的必要措施包括：在沙箱环境（Sandboxed Environment）中执行代码、实施最小权限原则、对敏感操作引入人工审核环节（Human-in-the-loop）。

采纳代码生成模式，意味着工程团队面临的挑战重心发生了根本性的转移。关注点不再是“模型会不会算错平均值？”，而是转化为一系列更为复杂的系统工程问题：1) **正确性（****Correctness****）**：LLM生成的代码在语法上是否正确？在逻辑上是否精确地反映了用户的真实意图？2) **安全性（****Security****）**：如何构建一个坚固的防护体系，以防止LLM生成并执行如DROP TABLE users;这样的恶意指令？3) **可观测性（****Observability****）**：当代码执行失败或返回了非预期的结果时，我们如何有效地调试从用户意图到LLM解读，再到代码生成的整个链条？这清晰地表明，将LLM用于生产级数据分析，已经不再是一个单纯的机器学习建模问题，而是一个关乎软件工程、系统安全和DevOps的综合性挑战。任何计划将此技术产品化的组织，都必须为应对这些工程挑战做好充分准备。

### 1.3 RAG框架：通过目标数据检索增强LLM能力

除了代码生成，检索增强生成（Retrieval-Augmented Generation, RAG）是另一个用于连接LLM与外部知识的基础模式。理解RAG的机制、优势及其局限性，对于构建一个全面的数据分析系统至关重要。

**核心概念**：RAG的核心思想是在LLM生成回答之前，先从一个外部知识库中**检索**与用户问题相关的信息，然后将这些检索到的信息作为上下文（Context）注入到LLM的提示（Prompt）中。这个过程能够有效地将模型的回答“锚定”在真实的数据上，从而显著减少内容幻觉，并使其能够获取并利用其训练截止日期之后的信息 6。

**在结构化数据中的应用**：将RAG应用于结构化数据，其工作流程通常如下：

1. **嵌入（****Embedding****）**：将结构化数据集中的每一行（或有意义的行组）通过一个嵌入模型（Embedding Model）转换成一个高维的数字向量。这个向量旨在捕捉该行数据的语义信息 6。

2. **索引（****Indexing****）**：将生成的所有向量存储在一个专门的向量数据库（Vector Database）中。向量数据库经过优化，能够进行高效的、基于向量相似度的搜索 6。

3. **检索（****Retrieval****）**：当用户提出一个问题时（例如，“告诉我关于客户ID 12345的失败支付记录”），系统首先将这个问题本身也转换成一个查询向量。然后，利用这个查询向量在向量数据库中进行搜索，找出与查询意图最相似的数据行向量。

4. **增强与生成（****Augmentation & Generation****）**：系统将检索到的最相关的数据行（通常是原始的文本或JSON格式）与用户的原始问题一起，组合成一个新的、内容丰富的提示，并发送给LLM。LLM最终基于这个被增强了的上下文来生成一个自然的、有事实依据的回答。

**优势分析**：

● RAG非常适合处理“查找型”（lookup-style）或对话式的查询，特别是当用户关心的是关于某个特定记录或实体的具体信息时 6。

● 由于LLM被强制要求基于提示中提供的真实数据进行回答，这极大地降低了它在回答关于某一笔具体交易时凭空捏造细节的可能性。

**在聚合分析场景下的关键局限性**：

● **无法获得全局视图**：RAG的设计初衷决定了它在每次查询时，只会检索并提供数据集中一个很小的子集（例如，相似度最高的5条或10条记录）。因此，它从根本上无法回答那些需要对整个数据集进行扫描和计算才能得出的问题，比如“我们公司**整体**的支付成功率是多少？” 6。

● **语义稀释问题**：当一个数据表的列（属性）非常多时，向量嵌入会面临“语义稀释”的挑战。每一行数据的向量是由其所有列的信息共同构成的。在这种情况下，单个列的特定值所占的“语义权重”会被稀释。例如，一个关于“年龄为18岁”的查询，可能会因为其他列中出现了“年轻”、“驾照”等语义上相关的词，而错误地检索到年龄并非18岁的行，从而影响检索的精确度 5。

用户最初的问题似乎是在不同方法之间进行选择。然而，一个真正先进的系统架构并非是“非此即彼”的选择，而是对多种方法的**协同编排**。设想一个更复杂的查询：“请比较我们的新功能‘QuickPay’与标准支付流程，在欧洲高价值客户群体中的支付成功率差异。” 一个成熟的智能体系统（Agentic System，详见第二部分）在处理这个查询时，会展现出更高层次的智能。它首先会启动一个类似RAG的流程（或使用命名实体识别技术 5）来

**解析和识别**查询中的关键实体：“QuickPay功能”、“高价值客户”、“欧洲”。这个过程可能需要查询一个内部的元数据存储库或产品文档，以将这些模糊的自然语言概念转化为精确的机器可读标识（例如，feature_id = 'QP_01', customer_tier = 'premium', region = 'EU'）。

一旦这些实体被成功解析为结构化的过滤条件，系统的下一个步骤就是将这些条件传递给一个**代码生成模块**。这个模块的任务不再是理解模糊的语言，而是基于这些清晰的、结构化的输入，构建一条精确的SQL查询（如 SELECT... WHERE feature_id = 'QP_01' AND customer_tier = 'premium' AND...），以执行最终的聚合分析。这个例子揭示了一个深刻的架构思想：RAG和代码生成并非相互排斥的方法，而是一个更宏大分析工作流中两个相辅相成的关键工具。RAG负责处理语义理解和实体解析（回答“是什么”的问题），而代码生成则负责执行精确的、大规模的量化计算（回答“有多少”的问题）。这种融合，代表了LLM在企业数据分析领域走向成熟的必经之路。

### 表1：面向结构化数据的LLM应用方法论对比分析

为了清晰地总结上述基础方法论，下表提供了一个直观的对比，旨在帮助决策者快速理解不同方法的适用场景、优势与劣势。

|   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|
|方法论|核心机制|理想查询类型|可扩展性|主要优势|关键弱点|支付成功率分析适用性|
|**直接提示** **(Baseline)**|直接将少量表格数据作为文本放入提示中，要求LLM进行分析和计算。|极简单的、小规模数据的问答，如“总结这5条交易记录”。|极差，受限于LLM的上下文窗口长度，无法处理大规模数据。|实现简单，无需额外架构。|容易产生计算幻觉，无法扩展，对数据格式敏感 2。|**不适用**。无法处理真实世界的支付日志规模，且计算结果不可靠。|
|**检索增强生成** **(RAG)**|将数据行嵌入向量数据库，根据查询检索最相关的几行数据作为上下文提供给LLM 6。|针对特定记录的查找和对话式问答，如“查询客户A的最后一笔失败交易原因”。|中等。可处理大型数据集的索引，但单次查询只能分析少量检索到的数据。|减少事实性幻觉，能处理训练集之外的数据，提供溯源依据。|无法进行全局聚合分析，对多属性查询的检索精度可能下降 5。|**适用于辅助任务**。可用于查询特定失败交易的详细信息，但无法计算整体成功率。|
|**代码生成** **(NL-to-SQL/Python)**|LLM根据自然语言查询和数据元数据，生成SQL或Python代码，交由外部引擎执行 6。|复杂的聚合分析、商业智能（BI）式查询，如“按国家和支付渠道计算Q3的支付成功率”。|极佳。分析能力仅受限于后端数据引擎的性能，可扩展至PB级数据。|分析能力强大、计算结果精确、可扩展性好。|生成的代码可能存在错误或逻辑漏洞，存在严重的安全风险（代码注入）5。|**高度适用**。是进行整体支付成功率分析、趋势洞察和多维度下钻的核心方法。|

---

## 第二部分：面向复杂性与规模的架构设计

在掌握了基础方法论之后，本部分将深入探讨在真实世界中处理大规模数据和复杂分析任务时所面临的更严峻的挑战。我们将直接回应用户关于如何突破上下文窗口限制的问题，并引入“智能体”（Agentic）这一前沿的架构范式，它已成为构建精密、多步骤分析系统的主流方向。

### 2.1 超越上下文窗口：管理大规模数据的高级策略

LLM的上下文窗口（Context Window）是其一次性能够处理的信息量的上限，以“令牌”（Token）为单位计算。当分析任务所需的上下文信息（无论是原始数据、文档还是对话历史）超过这个限制时，系统性能就会受到严重制约。然而，解决这个问题并非简单地扩大窗口尺寸，工程实践已经发展出一系列更为精妙的策略。

一个正在演变中的挑战是，即便模型拥有了极长的上下文窗口，其性能也可能因为提示中包含了过多不相关或具有误导性的信息（被称为“硬性负样本”，Hard Negatives）而下降。因此，工程目标已经从“如何塞入更多数据”转变为“如何高效地塞入**正确**的数据”，即最大化上下文中的信噪比 8。

以下是应对上下文窗口限制的一系列解决方案，从简单到复杂：

1. **初步方法及其缺陷**：

○ 截断（Truncation）：这是最直接的方法，即从文本的开头或结尾简单地剪掉一部分内容以满足长度限制。这是一种权宜之計，但其代价是可能随机地丢失关键信息，导致模型无法获得完整的上下文 9。

2. **精细化的****RAG****分块与检索策略**：

○ 语义分块（Semantic Chunking）：相较于按固定字符数或令牌数进行硬性切分，语义分块旨在文本的自然断点（如段落结尾、一个完整概念的结束处）进行切分。这种方法更能保持每个数据块内部的上下文完整性 9。选择何种分块策略，受到文本自身结构、所用嵌入模型的能力以及目标LLM上下文长度的共同影响 10。

○ 选择性检索与重排序（Selective Retrieval & Re-ranking）：在RAG流程中，可以先从向量数据库中初步检索出一个较大数量的候选数据块（例如，前50个），然后使用一个更轻量级的模型或算法对这些候选块进行相关性重排序，并过滤掉噪声信息，最后只将最相关的几个块发送给核心LLM 8。这种方法是解决“硬性负样本”问题的直接手段。

3. **上下文压缩与重构技术**：

○ 摘要链（Summarization Chains）：对于非常长的文档（例如一份详细的年度财报），可以将其分解为多个章节或部分，然后使用LLM对每个部分单独进行摘要。最后，再将这些摘要汇总起来，生成一个“摘要的摘要”。这种方式创建了一个信息的层级化、压缩化表示，能够在有限的窗口内传达核心内容 7。

○ 结构化JSON的迭代式提示填充（Iterative Prompt Stuffing with Structured JSON）：这是一种功能强大且与RAG不同的方法。系统将一个长文档按顺序切分成较大的块（例如，数万个令牌），然后依次处理。在处理第一个块时，LLM被要求提取关键信息并以结构化的JSON格式输出。在处理第二个块时，系统会将第一个块生成的JSON摘要与第二个块的原文一同放入提示中，让LLM在已知前文摘要的背景下处理新内容。这个过程不断迭代，最终形成一个包含了整个文档精华信息的、高度浓缩的最终JSON对象。该方法实现了对文档的**全覆盖**，且无需维护向量数据库，特别适用于需要完整理解文档的场景 12。

4. **任务分解（****“****分而治之****”****策略）**：

○ 对于需要处理多种不同类型输入或执行多个逻辑步骤的复杂任务，最佳实践是采用多提示方法。系统首先通过一个LLM调用来对用户的初始查询进行**分类**，然后根据分类结果将其**路由**到预设的、针对该类任务优化的特定提示或工作流中 7。例如，一个“销售咨询”会被路由到查询CRM系统的工具链，而一个“技术支持问题”则会被路由到查询知识库的工具链。这种模式是下一节将要讨论的多智能体系统的前身。

业界对“长上下文窗口”的追逐一度被认为是解决所有问题的银弹。然而，一个深刻的现实是，长上下文窗口本身并非万能药，它反而催生了新的设计挑战。研究明确指出，随着在提示中增加更多的检索文档，RAG系统的性能有时反而会**下降**，因为模型会被那些语义相似但事实错误或不相关的“硬性负样本”所干扰 8。这引出一个至关重要的结论：核心的工程挑战正在从“我们如何将数据塞进上下文”转变为“我们如何设计高效的检索、过滤和排序系统，以便在可用的上下文中提供最高的信噪比？”。问题演变成了一个信息论和相关性工程的挑战，而不仅仅是暴力的数据填充。这意味着，即使未来模型拥有无限的上下文窗口，像重排序、选择性检索和摘要这样的技术，仍将是构建高性能、高可靠性LLM应用的关键所在。

### 表2：克服上下文窗口限制的高级策略对比

下表对本节讨论的各种高级策略进行了分类和比较，为技术团队在架构选型时提供参考。

|                        |                                          |                                 |                              |                         |
| ---------------------- | ---------------------------------------- | ------------------------------- | ---------------------------- | ----------------------- |
| 策略                     | 核心机制                                     | 解决的关键问题                         | 主要优势                         | 实现复杂度/成本                |
| **语义分块**               | 根据文本的语义结构（如段落）进行切分，而非固定长度 9。             | 保持数据块内部的上下文完整性，避免关键信息被割裂。       | 检索到的上下文更有意义，有助于LLM更好地理解。     | 中等。需要更复杂的文本处理逻辑。        |
| **选择性检索与重排序**          | 初步检索大量候选块，再用一个轻量级模型或算法进行筛选和排序 11。        | 过滤掉不相关或误导性的信息（“硬性负样本”），提高信噪比 8。 | 显著提升最终生成内容的准确性和相关性。          | 中等偏高。增加了一次排序/过滤的计算开销。   |
| **摘要链**                | 将长文档分块，逐块摘要，最后对摘要进行再摘要，形成层级化信息 7。        | 处理远超上下文窗口长度的单个长文档。              | 能够以压缩形式保留整个文档的核心信息。          | 中等。需要多次LLM调用，延迟较高。      |
| **迭代式提示填充** **(JSON)** | 顺序处理文档块，每一步都生成结构化JSON摘要，并将其作为下一步的上下文 12。 | 在不丢失任何信息的前提下，完整处理超长文档。          | 实现全文档覆盖，无需向量数据库，输出为结构化数据 12。 | 高。流程是串行的，延迟高，且需要精心设计提示。 |
| **任务分解与路由**            | 用一个初始LLM调用对用户查询进行分类，然后路由到专门的工作流 7。       | 处理需要不同工具或逻辑路径的复合型任务。            | 提高系统的模块化和效率，每个工作流都可以被高度优化。   | 高。需要设计分类器和多个独立的子系统。     |

### 2.2 智能体架构的兴起：从单一工具到多智能体系统

随着LLM技术的发展，业界的应用范式正在经历一场深刻的变革：从将LLM仅仅视为一个被动响应的“工具”，转向设计能够自主“思考”、“规划”和“行动”的**智能体（Agent）**系统。

**智能体的定义**：一个AI智能体远不止是一个LLM。它是一个以LLM为核心“大脑”或“推理引擎”的完整系统。这个系统能够感知其环境（通过输入），创建并执行一个计划，并利用外部工具来完成指定的任务 14。这标志着从简单的单轮问答向持续的、多步骤的自动化任务执行的演进。

**吴恩达（****Andrew Ng****）的智能体设计模式**：作为人工智能领域的权威人物，吴恩达提出了一套极具影响力的智能体设计模式框架，为理解和构建智能体能力提供了清晰的指引 15。

● 反思（Reflection）：智能体能够审视和批判自己的输出，并在此基础上进行迭代改进。例如，一个编写代码的智能体可以被提示检查自己生成的代码是否存在bug或不符合规范之处，然后进行重写 19。这个看似简单的模式，却能带来惊人的性能提升。

● 工具使用（Tool Use）：为智能体提供一系列外部工具的访问权限（如网络搜索API、计算器、代码解释器、数据库查询引擎等），并让它自主决定在何时、如何使用这些工具来达成目标 15。这是连接LLM与真实世界数据和行动的基石。

● 规划（Planning）：智能体能将一个复杂的目标分解为一系列更小、更易于管理的步骤。像ReAct（Reason + Act）这样的技术，让LLM在执行前先“大声思考”其计划，展示其推理过程，从而提高任务执行的逻辑性和成功率 15。

● 多智能体协作（Multi-agent Collaboration）：这是最先进的模式。一个复杂的任务被分解并分配给多个具有专门角色的智能体，它们协同工作，通过沟通、协作甚至辩论来共同解决问题 17。

**AI****开发范式的转变**：吴恩达明确指出，未来AI能力的提升，将同等地来自于这些智能体工作流的进步和基础模型本身的迭代 15。这意味着，AI开发的重心正在从如何对单个模型进行提示工程（Prompt Engineering），转向如何设计这些复杂智能体系统的

**架构**。

这些设计模式并非相互孤立，而是可以**组合**的，它们构成了构建复杂AI能力的分层“乐高积木”。一个先进的多智能体协作系统 17，其内部的每一个独立智能体可能都在使用“工具使用” 15和“规划” 20模式来完成自己的子任务。整个系统可能还会采用一个“反思”模式，即设立一个“批评家”智能体（Critic Agent）来审查一个“工作者”智能体（Worker Agent）的产出 19。这种组合性提供了一条清晰的、可操作的采纳路径：一个组织可以从为一个单一任务实现一个简单的“反思”循环开始，逐步迭代，最终构建出一个功能强大的、完整的“多智能体协作”系统。这为企业将AI从实验品推向生产力工具提供了切实可行的演进蓝图。

### 2.3 对比分析：金融分析场景下的单智能体与多智能体工作流

在决定构建一个智能体系统时，一个核心的架构抉择是采用单智能体还是多智能体。这一选择直接关系到系统的复杂度、可靠性和可扩展性。

**单智能体系统（****Single-Agent System****）**：

● **架构**：一个单一的LLM作为中央控制器，系统为其提供一个“工具箱”，其中包含多种不同功能的工具（例如，数据库查询工具、数据可视化工具、文件读写工具等）。这个智能体需要根据用户的请求，自主决定在每一步调用哪个工具来完成任务 21。

● **优势**：

○ **简单快速**：设计和构建相对简单，对于不那么复杂的任务，执行速度更快，因为没有智能体之间的通信开销 21。

● **劣势**：

○ **存在复杂性上限**：随着工具数量和任务复杂度的增加，单个智能体很容易变得“困惑”，可能会选错工具，并且系统难以调试。管理所有工具所需的庞大而复杂的提示本身就成了一种负担 22。

○ **控制力较弱**：对工作流程的控制力较弱，模型的行为有时难以预测和引导 21。

**多智能体系统（****Multi-Agent System, MAS****）**：

● **架构**：一个复杂的任务被分解，并分配给一个由多个专业化智能体组成的“团队”。例如，一个“编排者”（Orchestrator）智能体接收用户查询并制定计划，一个“数据库智能体”负责查询数据，一个“分析智能体”负责处理数据，最后由一个“报告智能体”生成总结。智能体之间通过通信和协调来达成最终目标 22。

● **优势**：

○ 专业化与高质量：每个智能体角色单一、工具集有限，这使得它们的行为更可预测，执行任务的质量和可靠性更高 23。

○ 可扩展性与模块化：系统架构是模块化的，更容易根据需求变化来增加、移除或修改某个智能体，从而具有更好的可扩展性和可维护性 22。

○ 鲁棒性：系统具有更好的容错性。如果某个智能体出现故障，其他智能体有可能接管其任务，保证整个系统的持续运行 14。

○ 解决复杂问题：通过模拟人类团队的协作模式，MAS能够解决那些对单个智能体而言过于复杂的问题 25。

● **劣势**：

○ 极高的复杂性：设计、构建和编排的难度远超单智能体系统。需要仔细设计智能体间的通信协议、数据流和协作机制 21。

○ 通信开销：智能体之间的交互会增加系统的延迟和计算成本。这就像管理一个真实的人类团队，成员越多，沟通成本越高 16。

○ 潜在的混乱：如果没有一个结构化的、强有力的编排模式，让多个智能体自由交互很可能导致混乱、不可预测甚至完全错误的结果 27。研究表明，设计不佳的MAS的正确率可能非常低（甚至低至25%），并且会遭受“组织设计缺陷”的困扰，例如智能体之间互相忽略对方的输入、丢失对话历史等 28。

关于单智能体与多智能体的辩论，其核心触及了一个深刻的、近乎哲学性的问题。这不仅仅是一个技术选择，更是一个关于如何为AI设计**组织架构**的战略决策。争论的核心在于**上下文管理**：应该将所有上下文信息集中在一个“大脑袋”里（单智能体），还是将其分散给各个领域的“专家”（多智能体）？27

这完美地映射了人类社会的组织结构演变。一个初创公司（类比单智能体）行动敏捷、沟通成本低，但难以扩展处理复杂业务。一个大型企业（类比多智能体）能力强大、分工明确，但可能因层级和流程而变得缓慢，并饱受部门壁垒之苦。对MAS失败模式的研究 28揭示的问题——如“信息隐瞒”、“忽略他人输入”——听起来更像是组织功能失调，而不仅仅是代码bug。

因此，构建一个成功的MAS，挑战已超越了机器学习本身，进入了**系统设计**和**组织理论**的范畴。开发者需要像设计一个高效的人类组织一样，去思考智能体的角色、职责、汇报关系和沟通渠道。这预示着未来AI开发所需技能的深刻转变，从单纯的算法工程师，向兼具系统架构师和AI组织设计师能力的复合型人才演进。

### 表3：单智能体与多智能体系统架构对比

下表为决策者提供了一个战略性的比较，帮助判断何时需要从简单的单智能体系统“扩展”到更复杂的多智能体架构。

|   |   |   |   |
|---|---|---|---|
|属性|单智能体系统|多智能体系统|何时选择|
|**架构模式**|单一LLM作为中央控制器，管理一个工具箱 21。|分布式，多个专业化智能体协同工作，通常由一个编排者协调 22。|-|
|**适用任务复杂度**|适用于简单、线性的任务，或作为复杂系统的原型。|适用于需要分解、多步骤、涉及不同专业领域的复杂任务 23。|**单智能体**：快速原型验证，工作流清晰。**多智能体**：任务需要明确的责任分离和专业化。|
|**可扩展性**|有限。随着工具和功能增多，系统变得笨重且难以维护 22。|高。模块化设计，易于添加、修改或替换单个智能体 24。|**单智能体**：工具集较小且范围固定。**多智能体**：预计未来功能会持续扩展，工具种类繁多。|
|**开发成本****/****精力**|较低。设计和实现相对直接。|较高。需要精心设计通信、协作和状态管理机制 21。|**单智能体**：预算和时间有限。**多智能体**：有充足的资源进行复杂的系统设计和调试。|
|**可靠性****/****调试**|相对容易调试单个智能体的行为，但容易出现选错工具的错误。|单个智能体行为可控，但系统级的交互和协作问题难以调试。|**单智能体**：对任务执行的精确顺序要求不高。**多智能体**：对任务执行的可靠性和质量有极高要求。|
|**关键失败模式**|提示过载、工具选择混淆、任务处理走捷径 21。|智能体间协调失败、信息传递丢失、任务偏离、过早终止 28。|-|

---

## 第三部分：行业先锋：金融服务领域的先进实践

本部分将理论付诸实践，通过深入分析三家代表性企业的案例，具体回答用户关于“业界先进实现”的问题。这三家公司——摩根大通、彭博社和Databricks——分别代表了在金融和科技领域应对LLM数据分析挑战的三种截然不同但均获成功的战略路径。

### 3.1 案例研究：摩根大通（J.P. Morgan）—— 安全合规驱动下的自建AI生态

摩根大通的案例展示了一种“自建围墙花园”（Walled Garden）的模式。在顶级投资银行极度严格的安全和监管要求驱动下，他们选择了一条自主研发、完全可控的道路。

**核心战略**：出于对数据隐私和安全的极度审慎，摩根大通已明确禁止内部使用ChatGPT等外部生成式AI工具 29。其核心战略是在其先进的内部数据平台“JADE”（JPMorgan Chase Advanced Data Ecosystem）之上，构建一个完全专有的、端到端的AI生态系统 30。

**关键组件**：

● JADE平台：这是整个AI战略的基石。JADE提供了一个高质量、统一且受严格治理的数据环境，为AI模型的训练和部署提供了必要的基础 30。它采用数据网格（Data Mesh）架构，将数据所有权下放到各个业务产品线，从而加速了数据的访问和开发迭代。

● LLM Suite：这是一个在内部开发并已投入生产的“虚拟研究分析师”平台。它已被推广给数万名员工使用，用于自动化处理如总结复杂金融文件、起草投资备忘录、从海量数据集中提炼洞察等任务 29。

● 配套工具生态：LLM Suite并非孤立存在，而是其更广泛AI生态的一部分。该生态还包括用于提供合规指导的Connect Coach和用于进行实时市场分析的SpectrumGPT等工具，它们共同构成了一个强大的AI基础设施 29。

**架构哲学**：

● 人类增强，而非替代：摩根大通明确其AI的目标是**增强**人类分析师的专业能力，而非取代他们。AI工具旨在消除重复性的操作摩擦，让专业人士能专注于更高层次的战略思考和决策 29。

● 代码生成驱动分析：在数据分析的具体实现上，其方法与本报告1.2节描述的“代码生成”模式高度一致。LLM被用作自然语言接口，将用户的查询翻译成可以在专用分析引擎上执行的代码或指令，从而让非技术用户也能轻松地查询和分析数据 31。

● 安全与治理先行：整个技术栈的设计都将合规性放在首位。从底层的数据平台（JADE）到上层的AI治理框架（Infinite AI），后者负责管理模型的整个生命周期，确保所有AI应用都符合内外部的监管要求 30。

**强大的内部研究能力**：摩根大通的AI研究部门（AI Research program）持续在可解释性AI、合成数据生成以及评估长上下文LLM在金融概念理解方面的性能等前沿领域发表高质量研究成果，这表明了其在基础能力上的长期和深度投入 32。

对于像摩根大通这样的大型、受严格监管的金融机构而言，AI架构的设计已远非一个单纯的技术选型问题，它已成为其整体**风险管理和数据治理战略**不可分割的一部分。他们为何要投入巨资自建那些理论上可以通过API租用的能力？答案并非仅仅为了追求更高的性能。答案是**控制**。通过完全的内部研发 29，他们能够牢牢控制从数据接入、模型训练、推理服务到治理流程的每一个环节。其精心设计的架构 30不仅是一个技术栈，更是一道坚固的

**合规与安全护城河**。这对任何处于类似监管环境下的公司（例如支付行业）都具有深刻的启示：选择使用外部模型还是自建模型，首要的考量因素可能不是技术优劣，而是风险和合规。整个系统架构必须能够被审计，并且在监管机构面前具有可辩护性，而这在一个完全专有的系统中显然更容易实现。

### 3.2 案例研究：彭博GPT（BloombergGPT）—— 领域知识预训练的力量

彭博社的案例代表了另一种强大的战略——“数据护城河”（Data Moat）。当一个公司拥有独特、高质量的专有数据时，它可以选择构建一个高度专业化的模型，使其在特定领域内的表现超越所有通用模型。

**核心战略**：彭博社做出了一个重大的战略决策：从零开始训练一个拥有500亿参数的大语言模型。他们坚信，其积累了数十年的独特金融数据是一项无可比拟的核心资产，能够训练出最懂金融的AI 4。

**关键资产：数据**：BloombergGPT与其他模型的最大区别在于其训练语料库。他们构建了一个超过7000亿令牌的庞大数据集，其中约一半是其专有的、经过精心整理的FinPile——包含了自2007年以来积累的金融报告、新闻、公司文件、财报电话会议记录等高质量金融文本。另一半则来自通用的公共数据集，以保证模型具备广泛的常识能力 35。

**独特的架构设计**：

● 模型主体基于BLOOM的仅解码器（Decoder-only）Transformer架构 36。

● 一个至关重要的架构细节是其**定制化的分词器**。彭博的工程师敏锐地意识到，标准分词器在处理数字上的缺陷对于金融领域是致命的。因此，他们设计了一个特殊的分词器，该分词器会将数字拆分成单个的数字字符（例如，'5234' -> '5', '2', '3', '4'），这有助于模型更好地学习数字的结构和数值关系 4。此外，其词汇表规模也远大于通用模型（13.1万 vs. 普遍的5万），以便更有效地编码金融领域的专业术语 35。

**卓越的成果**：实验结果表明，BloombergGPT在各类金融领域的专业基准测试中，其性能显著优于规模相似甚至更大的通用模型，同时在通用任务上的表现也毫不逊色 36。这雄辩地证明了领域知识预训练的巨大价值。

**核心应用场景**：一个典型的应用是将自然语言翻译成彭博查询语言（Bloomberg Query Language, BQL）。这完美地印证了本报告1.2节中描述的“代码生成”模式，只不过是针对其专有数据生态系统的高度定制化版本 4。

对模型而言，领域知识预训练是**终极形式的****“****提示工程****”**。它创造了一个拥有内置的、无法被轻易擦除的领域知识的模型。微调（Fine-tuning）和RAG是在推理时向一个“通才”模型**注入**知识的手段，而领域预训练则是将这些知识**固化**在模型的根本参数之中。这使得模型的整个“世界观”都是由金融数据塑造的。其定制化的分词器 4就是一个绝佳的例子——这个模型“看到”数字的方式从根本上就与通用模型不同。这意味着，对于拥有独特数据和术语体系的核心业务领域（例如支付处理），一个规模较小但经过领域数据专门训练的模型，其表现可能远超一个规模庞大但泛泛而谈的通用模型（如GPT-4），同时还可能具备更低的成本和更高的可靠性。这为企业提供了一个关键的战略抉择：是租用一个“通才巨人”，还是打造一个“专家精英”？

### 3.3 案例研究：Databricks湖仓一体—— 企业级GenAI的统一平台

Databricks的案例代表了第三种战略路径——“赋能者”或“平台”战略。它为广大企业提供了构建自己AI解决方案所需的全套工具，使它们不必从零开始，也能享受到先进AI技术带来的红利。

**核心战略**：Databricks提供了一个统一的数据与AI平台，其核心价值在于使企业能够灵活地实施所有四种主流的生成式AI架构模式：提示工程、RAG、微调和预训练 38。

**平台架构**：

● 湖仓一体（The Lakehouse）：这是平台的基石，它统一了数据仓库和数据湖，使得企业可以将所有类型的数据（结构化、半结构化、非结构化）存储和管理在一个地方，为AI应用提供了统一的数据源 39。

● Unity Catalog：这是一个统一的治理层，能够对所有数据和AI资产（包括模型、Notebook、仪表盘等）进行统一的权限管理、血缘追踪和审计。这对于金融等需要严格合规的行业至关重要 39。

● Mosaic AI：这是一套用于构建、训练和部署AI模型的工具集，其中包括了用于微调和从头预训练自定义LLM的强大功能 41。

**面向金融服务的核心能力**：

● **灵活性**：客户可以自由选择使用流行的开源模型（如Llama系列、MPT）、通过API调用专有模型（如OpenAI），或者利用平台能力训练完全属于自己的模型 41。

● **高质量****RAG**：Databricks提供了一整套用于构建和优化RAG应用的工具，涵盖了从数据准备、向量搜索、模型排序到监控的全过程 38。

● **高效的模型训练**：平台提供了如Mosaic AI Pretraining这样的解决方案，帮助企业以更低的成本、更短的时间构建自己的定制化LLM 41。

**旗舰模型****DBRX**：Databricks自己推出的开源模型DBRX，是现代LLM架构的一个典范。它采用了**细粒度的专家混合（Mixture-of-Experts, MoE）**架构，总参数量为1320亿，但在处理任何输入时，只有360亿参数是活跃的。这种设计使得其在推理时的速度和成本效率远高于同等规模的密集型（Dense）模型，同时在性能上达到了业界顶尖水平 40。

企业AI的未来，关键点已不再是选择某一个特定的“最佳模型”，而是选择一个能够提供**灵活性、可扩展性和强大治理能力**的**数据与****AI****平台**。摩根大通和彭博社的案例展示了拥有海量资源的企业可以达到的高度，但这对于大多数公司而言是难以复制的。Databricks的战略 38正是抓住了这一市场现实。它的核心价值主张是提供一个底层的AI“工厂”（即湖仓一体平台），让企业能够在这个工厂里，安全、合规地构建属于自己的、满足特定业务需求的解决方案。而其DBRX模型的MoE架构 42则是这个战略中至关重要的一环。它通过大幅降低高性能AI的推理成本——这往往是企业规模化应用的最大障碍——从而实现了先进AI能力的“民主化”。这预示着，对许多公司而言，最终的制胜策略将不是押注于某一种单一的模型架构，而是投资于一个能够让他们适应技术变革、组合不同模式并持续演进的强大平台。

---

## 第四部分：战略综合与未来展望

本报告的最后一部分将对前文的分析进行全面综合，为用户提供一个直接的、可操作的决策框架，并对该技术领域的未来发展趋势进行展望。

### 4.1 决策框架：为支付成功率分析选择正确的架构

本节旨在将报告中的技术洞察转化为一个清晰的、可操作的决策框架，帮助用户根据其具体需求，为支付成功率分析任务选择最合适的系统架构。

**核心任务回顾**：用户的目标是分析大规模、结构化的支付交易日志，以理解成功率的变化。这项任务既需要进行**聚合分析**（例如，“按国家统计的总体成功率”），也需要具备**下钻**到特定群体或单个交易进行深入探究的能力。

**需求与架构的映射**：下方的决策矩阵（表4）将指导架构的选择。

● **如果主要需求是生成聚合报告和进行商业智能（****BI****）风格的查询**，那么最稳健的起点是采用**代码生成**模式。

● **如果需要以对话方式查询特定交易的详细信息**，那么必须在系统中集成**RAG**组件。

● **如果查询变得高度复杂且涉及多个步骤**，那么系统应从一个简单的脚本演进为一个能够进行规划并同时使用RAG和代码生成工具的**单智能体系统**。

● **如果整个工作流涉及多个逻辑上截然不同的角色（如数据提取、欺诈检测、数据分析、报告生成），并且对可靠性有极高要求**，那么最终的理想架构将是一个**多智能体系统**。

支付成功率分析的推荐架构：

基于上述分析，一个理想的、面向未来的支付成功率分析系统，应采用一个由“规划者”智能体（Planner Agent）编排的多智能体系统。其工作流程示例如下：

1. **接收用户查询**：用户输入：“为什么上周我们在德国的PayPal交易支付成功率下降了？”

2. **规划者智能体接收与分解**：中心的“规划者”智能体接收此查询。它利用其规划能力，将这个复杂的、模糊的问题分解为一系列清晰的、可执行的子任务。

3. **任务分配**：

○ 任务1：“识别相关的数据库表和字段。” -> 分配给**数据库智能体**（Database Agent），该智能体拥有访问数据库元数据的权限。

○ 任务2：“精确定义‘上周’的时间范围，并构建针对‘德国’和‘PayPal’的过滤条件。” -> 分配给**代码生成智能体**（Code-Gen Agent）。

○ 任务3：“计算符合条件的本周与上周的支付成功率。” -> 继续由**代码生成智能体**执行，它将生成并运行相应的SQL查询。

○ 任务4：“查询内部知识库，查找上周是否有任何与德国PayPal相关的已知服务中断或事件报告。” -> 分配给一个**RAG****智能体**，该智能体专门用于查询内部的非结构化文档（如事件报告、运维日志）。

4. **执行与协作**：各个智能体并行或串行地执行自己的任务。代码生成智能体生成并运行SQL，返回结构化的数据结果（成功率数字）。RAG智能体则从文档中检索到相关的文本片段。

5. **综合与生成**：最后，一个**综合智能体**（Synthesis Agent）接收来自代码生成智能体的结构化数据和来自RAG智能体的非结构化上下文，将两者融合，最终生成一段全面的、通俗易懂的自然语言回答，向用户解释成功率下降的可能原因，并附上数据支持。

**模型选择考量**：考虑到金融场景对数字处理精度和专业术语理解的高要求，一个在金融数据上表现优异且拥有定制化分词器的模型（如BloombergGPT背后的设计理念 4）将是理想选择。如果从零开始预训练不可行，那么在内部支付数据上对一个强大的开源模型（如具有高效MoE架构的DBRX 42）进行微调，将是一个极具竞争力的替代方案。

### 表4：支付分析LLM架构决策框架

下表为用户提供了一个直接的、可操作的工具，通过将业务和技术约束与报告中详细介绍的架构选项直接关联，来辅助其进行战略决策。

|   |   |   |   |   |   |
|---|---|---|---|---|---|
|决策因素|简单代码生成脚本|仅RAG系统|单智能体 (RAG + Code-Gen)|多智能体系统|定制预训练模型|
|**主要查询类型**|聚合分析|记录查找|混合查询|复杂、多步骤、需要推理的查询|所有类型，但在特定领域表现更佳|
|**数据规模**|极佳|中等|极佳|极佳|极佳|
|**延迟要求**|低|中|中高|高|取决于模型大小和架构|
|**开发预算****/****时间**|低|中|中|高|非常高|
|**团队技能要求**|SQL/Python|向量数据库, RAG|Agent框架, Prompt工程|系统设计, Agent编排, 通信协议|深度学习, 分布式训练|
|**治理****/****审计需求**|中等|高（需追踪检索源）|高|非常高（需追踪Agent间交互）|非常高（完全控制数据和模型）|
|**推荐场景**|**起点**。用于快速实现核心的BI报表和聚合分析。|用于构建客服或运维的辅助查询工具。|**演进方向**。当需要在一个统一界面中同时处理聚合和查找任务时。|**最终目标**。当分析工作流需要多个专业角色协作，且对可靠性要求极高时。|当拥有独特数据护城河，且通用模型无法满足精度和专业性要求时的战略投资。|

### 4.2 结论分析：LLM驱动的金融智能之未来

本报告通过对LLM在结构化数据分析领域的基础方法、高级架构和行业实践进行系统性剖析，旨在为在金融科技领域探索AI应用提供一个清晰的路线图。综合所有分析，可以得出以下核心结论及对未来的展望：

**单一****AI****的终结，复合系统的崛起**：试图用一个单一的、庞大的LLM解决所有复杂问题的时代正在结束。未来属于由多个、功能专一的智能体、工具和模型组合而成的**复合****AI****系统**（Compound AI Systems）17。对于金融分析这类任务，这意味着系统将不再是一个“模型”，而是一个由数据查询智能体、分析智能体、合规检查智能体和报告生成智能体等组成的协同工作网络。

**数据与平台的至高重要性**：随着强大的基础模型通过开源或API的形式变得越来越普及，企业在AI时代的核心竞争力将来源于两个方面：

1. **其专有数据的质量和独特性**。这是彭博社案例（BloombergGPT）给予我们的深刻启示 4。拥有别人没有的高质量数据，是训练出卓越领域模型的根本。

2. **其底层数据与****AI****平台的稳健性和治理能力**。这是摩根大通和Databricks案例共同揭示的真理 30。一个能够统一管理数据、保障安全合规、并能灵活支持多种AI架构模式的平台，是企业AI战略成功的基石。

**AI****开发者角色的演进**：未来对AI开发人才的要求正在发生深刻转变，从单纯的模型训练和算法调优，转向**AI****系统架构设计**和**AI****组织设计**。成功的挑战不再仅仅是提升单个模型的准确率，而是如何编排这些复杂的、由AI构成的系统，如何管理它们之间的交互，并确保整个系统是可靠、安全且与业务目标高度一致的 28。

**最终建议**：对于用户提出的支付成功率分析目标，前进的道路是清晰的。它始于一个坚实的数据基础和一个安全的**代码生成**模式，以解决核心的聚合分析需求。随后，它应逐步演进为一个能够进行推理、分解任务、并综合多源信息的**多智能体系统**。这并非一个简单的项目，而是一项战略能力的构建。一旦正确建成，它将在日益由AI驱动的金融竞争格局中，为企业提供一种持久的、难以被复制的竞争优势。

#### 引用的著作

1. Unveiling Challenges for LLMs in Enterprise Data Engineering - arXiv, 檢索日期：6月 16, 2025， [https://arxiv.org/html/2504.10950v1](https://arxiv.org/html/2504.10950v1)

2. Improving LLM understanding of structured data and exploring ..., 檢索日期：6月 16, 2025， [https://www.microsoft.com/en-us/research/blog/improving-llm-understanding-of-structured-data-and-exploring-advanced-prompting-methods/](https://www.microsoft.com/en-us/research/blog/improving-llm-understanding-of-structured-data-and-exploring-advanced-prompting-methods/)

3. Fashion Forward Friday Why LLMs Struggle with Structured Data - The Ink Kitchen, 檢索日期：6月 16, 2025， [https://inkkitchen.com/fashion-forward-friday-why-llms-struggle-with-structured-data/](https://inkkitchen.com/fashion-forward-friday-why-llms-struggle-with-structured-data/)

4. BloombergGPT is Live. A Custom Large Language Model for Finance | Belitsoft, 檢索日期：6月 16, 2025， [https://belitsoft.com/bloomberggpt](https://belitsoft.com/bloomberggpt)

5. Why do LLMs struggle to understand structured data from relational databases, even with RAG? How can we bridge this gap? - Reddit, 檢索日期：6月 16, 2025， [https://www.reddit.com/r/LLMDevs/comments/1ixa80j/why_do_llms_struggle_to_understand_structured/](https://www.reddit.com/r/LLMDevs/comments/1ixa80j/why_do_llms_struggle_to_understand_structured/)

6. LLMs For Structured Data - neptune.ai, 檢索日期：6月 16, 2025， [https://neptune.ai/blog/llm-for-structured-data](https://neptune.ai/blog/llm-for-structured-data)

7. Techniques for overcoming context length limitations | IBM watsonx, 檢索日期：6月 16, 2025， [https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-context-length.html?context=wx](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-context-length.html?context=wx)

8. Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG | OpenReview, 檢索日期：6月 16, 2025， [https://openreview.net/forum?id=oU3tpaR8fm¬eId=8X6xAgSGa2](https://openreview.net/forum?id=oU3tpaR8fm&noteId=8X6xAgSGa2)

9. 5 Approaches to Solve LLM Token Limits - Deepchecks, 檢索日期：6月 16, 2025， [https://www.deepchecks.com/5-approaches-to-solve-llm-token-limits/](https://www.deepchecks.com/5-approaches-to-solve-llm-token-limits/)

10. Mastering RAG: Advanced Chunking Techniques for LLM Applications - Galileo AI, 檢索日期：6月 16, 2025， [https://www.galileo.ai/blog/mastering-rag-advanced-chunking-techniques-for-llm-applications](https://www.galileo.ai/blog/mastering-rag-advanced-chunking-techniques-for-llm-applications)

11. Understanding RAG Part V: Managing Context Length - MachineLearningMastery.com, 檢索日期：6月 16, 2025， [https://machinelearningmastery.com/understanding-rag-part-v-managing-context-length/](https://machinelearningmastery.com/understanding-rag-part-v-managing-context-length/)

12. RAG vs. Prompt Stuffing: Overcoming Context Window Limits for ..., 檢索日期：6月 16, 2025， [https://www.spyglassmtg.com/blog/rag-vs.-prompt-stuffing-overcoming-context-window-limits-for-large-information-dense-documents](https://www.spyglassmtg.com/blog/rag-vs.-prompt-stuffing-overcoming-context-window-limits-for-large-information-dense-documents)

13. Challenges in Structured Document Data Extraction at Scale with LLMs - Zilliz blog, 檢索日期：6月 16, 2025， [https://zilliz.com/blog/challenges-in-structured-document-data-extraction-at-scale-llms](https://zilliz.com/blog/challenges-in-structured-document-data-extraction-at-scale-llms)

14. LLM-based Multi-Agent Systems: Techniques and Business Perspectives - arXiv, 檢索日期：6月 16, 2025， [https://arxiv.org/html/2411.14033v2](https://arxiv.org/html/2411.14033v2)

15. Introduction to AI Agents - BRAINYX, 檢索日期：6月 16, 2025， [https://brainyx.co/journal/journal29/](https://brainyx.co/journal/journal29/)

16. Why the Future is Agentic: An Overview of Multi-Agent LLM Systems - Alexander Thamm, 檢索日期：6月 16, 2025， [https://www.alexanderthamm.com/en/blog/multi-agent-llm-systems/](https://www.alexanderthamm.com/en/blog/multi-agent-llm-systems/)

17. Exploring multi-agent AI systems | Generative-AI – Weights & Biases - Wandb, 檢索日期：6月 16, 2025， [https://wandb.ai/byyoung3/Generative-AI/reports/Exploring-multi-agent-AI-systems---VmlldzoxMTIwNjM5NQ](https://wandb.ai/byyoung3/Generative-AI/reports/Exploring-multi-agent-AI-systems---VmlldzoxMTIwNjM5NQ)

18. Four AI Agent Strategies That Improve GPT-4 and GPT-3.5 Performance - DeepLearning.AI, 檢索日期：6月 16, 2025， [https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/)

19. One Agent For Many Worlds, Cross-Species Cell Embeddings, and more - DeepLearning.AI, 檢索日期：6月 16, 2025， [https://www.deeplearning.ai/the-batch/issue-242/](https://www.deeplearning.ai/the-batch/issue-242/)

20. neural-maze/agentic-patterns-course: Implementing the 4 agentic patterns from scratch - GitHub, 檢索日期：6月 16, 2025， [https://github.com/neural-maze/agentic-patterns-course](https://github.com/neural-maze/agentic-patterns-course)

21. Agentic AI: Single vs Multi-Agent Systems | Towards Data Science, 檢索日期：6月 16, 2025， [https://towardsdatascience.com/agentic-ai-single-vs-multi-agent-systems/](https://towardsdatascience.com/agentic-ai-single-vs-multi-agent-systems/)

22. Multi-Agent AI Systems: When to Expand From a Single Agent - WillowTree Apps, 檢索日期：6月 16, 2025， [https://www.willowtreeapps.com/craft/multi-agent-ai-systems-when-to-expand](https://www.willowtreeapps.com/craft/multi-agent-ai-systems-when-to-expand)

23. Multi agent LLM systems: GenAI special forces - K2view, 檢索日期：6月 16, 2025， [https://www.k2view.com/blog/multi-agent-llm/](https://www.k2view.com/blog/multi-agent-llm/)

24. The Power of Multi-Agent Systems vs Single Agents - Relevance AI, 檢索日期：6月 16, 2025， [https://relevanceai.com/blog/the-power-of-multi-agent-systems-vs-single-agents](https://relevanceai.com/blog/the-power-of-multi-agent-systems-vs-single-agents)

25. Single-Agent vs Multi-Agent Systems: Two Paths for the Future of AI | DigitalOcean, 檢索日期：6月 16, 2025， [https://www.digitalocean.com/resources/articles/single-agent-vs-multi-agent](https://www.digitalocean.com/resources/articles/single-agent-vs-multi-agent)

26. Multi-Agent Collaboration Mechanisms: A Survey of LLMs - arXiv, 檢索日期：6月 16, 2025， [https://arxiv.org/html/2501.06322v1](https://arxiv.org/html/2501.06322v1)

27. Multi-Agent or Single Agent? : r/AI_Agents - Reddit, 檢索日期：6月 16, 2025， [https://www.reddit.com/r/AI_Agents/comments/1lb0zb3/multiagent_or_single_agent/](https://www.reddit.com/r/AI_Agents/comments/1lb0zb3/multiagent_or_single_agent/)

28. Why Do Multi-Agent LLM Systems Fail? - arXiv, 檢索日期：6月 16, 2025， [https://arxiv.org/pdf/2503.13657?](https://arxiv.org/pdf/2503.13657)

29. Next Gen AI in Action: How JPMorgan Chase's LLM Suite is Revolutionizing Financial Research - Global Skill Development Council, 檢索日期：6月 16, 2025， [https://www.gsdcouncil.org/blogs/next-gen-ai-in-action-how-jpmorgan-chase-s-llm-suite-is-revolutionizing-financial-research](https://www.gsdcouncil.org/blogs/next-gen-ai-in-action-how-jpmorgan-chase-s-llm-suite-is-revolutionizing-financial-research)

30. Twimbit AI Spotlight: J.P. Morgan Chase, 檢索日期：6月 16, 2025， [https://content.twimbit.com/insights/twimbit-ai-spotlight-j-p-morgan-chase/](https://content.twimbit.com/insights/twimbit-ai-spotlight-j-p-morgan-chase/)

31. The AI revolution for payments & tech | J.P. Morgan, 檢索日期：6月 16, 2025， [https://www.jpmorgan.com/payments/payments-unbound/volume-3/smart-money](https://www.jpmorgan.com/payments/payments-unbound/volume-3/smart-money)

32. Artificial Intelligence Research - J.P. Morgan, 檢索日期：6月 16, 2025， [https://www.jpmorgan.com/technology/artificial-intelligence](https://www.jpmorgan.com/technology/artificial-intelligence)

33. Machine Learning Center of Excellence (MLCOE) - J.P. Morgan, 檢索日期：6月 16, 2025， [https://www.jpmorgan.com/technology/applied-ai-and-ml/machine-learning](https://www.jpmorgan.com/technology/applied-ai-and-ml/machine-learning)

34. [2303.17564] BloombergGPT: A Large Language Model for Finance - arXiv, 檢索日期：6月 16, 2025， [https://arxiv.org/abs/2303.17564](https://arxiv.org/abs/2303.17564)

35. Paper Review: BloombergGPT: A Large Language Model for Finance - Andrey Lukyanenko, 檢索日期：6月 16, 2025， [https://andlukyane.com/blog/paper-review-bloomberggpt](https://andlukyane.com/blog/paper-review-bloomberggpt)

36. BloombergGPT: A Large Language Model for Finance - arXiv, 檢索日期：6月 16, 2025， [https://arxiv.org/html/2303.17564v3](https://arxiv.org/html/2303.17564v3)

37. Bloomberg & JHU's BloombergGPT: 'A Best-in-Class LLM for Financial NLP' | Synced, 檢索日期：6月 16, 2025， [https://syncedreview.com/2023/04/04/bloomberg-jhus-bloomberggpt-a-best-in-class-llm-for-financial-nlp/](https://syncedreview.com/2023/04/04/bloomberg-jhus-bloomberggpt-a-best-in-class-llm-for-financial-nlp/)

38. Generative AI Architecture Patterns - Databricks, 檢索日期：6月 16, 2025， [https://www.databricks.com/product/machine-learning/build-generative-ai](https://www.databricks.com/product/machine-learning/build-generative-ai)

39. Financial Services Solutions | Databricks Platform, 檢索日期：6月 16, 2025， [https://www.databricks.com/solutions/industries/financial-services](https://www.databricks.com/solutions/industries/financial-services)

40. Unlocking the Power of Generative AI with Databricks - Capitalize Analytics, 檢索日期：6月 16, 2025， [https://capitalizeconsulting.com/unlocking-the-power-of-gen-ai-with-databricks/](https://capitalizeconsulting.com/unlocking-the-power-of-gen-ai-with-databricks/)

41. Large Language Models - Databricks, 檢索日期：6月 16, 2025， [https://www.databricks.com/product/machine-learning/large-language-models](https://www.databricks.com/product/machine-learning/large-language-models)

42. Introducing DBRX: A New State-of-the-Art Open LLM | Databricks Blog, 檢索日期：6月 16, 2025， [https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)
