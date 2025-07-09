async def forward_164(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Survey and catalog established homogeneous organometallic catalyst systems for high-density polyethylene from ethylene, detailing metal center, ligand type, activator, operating conditions, and polymer properties."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, surveying HDPE catalysts, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    cot_sc_instruction = "Sub-task 2: Survey homogeneous organometallic catalyst systems that enable controlled chain walking using only ethylene, detailing metal center, ligand architecture, required activator class, branching frequency and regularity, and industrial deployment."
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask_1", "answer of subtask_1"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, surveying chain walking catalysts, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content] = thinking2_i
        answermapping[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    debate_instruction = "Sub-task 3: Conduct a structured debate for each statement (A-D) presenting pro and con arguments grounded in evidence from Subtasks 1 and 2, citing catalysts or industrial examples, and summarizing if each statement is supported or refuted."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds = self.max_round
    all_thinking3 = [[] for _ in range(N_rounds)]
    all_answer3 = [[] for _ in range(N_rounds)]
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": debate_instruction, "context": ["user query", "answer of subtask_1", "answer of subtask_2"], "agent_collaboration": "Debate"}
    for r in range(N_rounds):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking3_i, answer3_i = await agent([taskInfo, thinking1, answer1, thinking2, answer2], debate_instruction, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3_i, answer3_i = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
            all_thinking3[r].append(thinking3_i)
            all_answer3[r].append(answer3_i)
    final_debate_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_debate_agent([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Summarize debate outcomes for statements A-D with support or refutation.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_debate_agent.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    verif_instruction = "Sub-task 4: Verify key facts from the debate summary, resolve discrepancies, and select the single correct statement (A-D) regarding feasibility of regularly branched polyethylene from ethylene using a dual catalyst system."
    verif_agent = LLMAgentBase(["checking", "decision"], "Verification Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": verif_instruction, "context": ["user query", "answer of subtask_3"], "agent_collaboration": "Verification"}
    thinking4, answer4 = await verif_agent([taskInfo, thinking3, answer3], verif_instruction, is_sub_task=True)
    agents.append(f"Verification Agent {verif_agent.id}, verifying facts and selecting correct statement, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs