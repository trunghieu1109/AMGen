async def forward_171(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    # Stage 1
    cot_sc_instruction = "Sub-task 1: Based on the query, extract and summarize the observed population ratio N₁/N₂=2, the energy difference ΔE=1.38×10^-23 J, and the LTE assumption for both stellar photospheres."
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N1):
        thinking1_i, answer1_i = await cot_agents1[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, summarizing data, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr1 = "Given all the above thinking and answers, find the most consistent summary of the physical data for the problem."
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings1 + possible_answers1, final_instr1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1, "answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Stage 2
    cot_sc_instruction2 = "Sub-task 2: Apply the Boltzmann distribution in LTE to write the ratio of excited-level populations N₁/N₂ for star_1 and star_2 and take its natural logarithm, using context from the summary."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction2,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"SC_CoT"}
    for i in range(N2):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, deriving ln(N₁/N₂) expression, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings2.append(thinking2_i)
        possible_answers2.append(answer2_i)
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr2 = "Given all the above thinking and answers, find the most consistent logarithmic expression for the population ratio."
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, final_instr2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2, "answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Stage 3
    cot_sc_instruction3 = "Sub-task 3: Rearrange the expression ln(N₁/N₂)=ΔE/k*(1/T₂−1/T₁) to isolate ln(2) purely as a function of T₁ and T₂."
    N3 = self.max_sc
    cot_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_sc_instruction3,"context":["user query","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"SC_CoT"}
    for i in range(N3):
        thinking3_i, answer3_i = await cot_agents3[i]([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents3[i].id}, isolating ln2, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_thinkings3.append(thinking3_i)
        possible_answers3.append(answer3_i)
    final_decision_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr3 = "Given all the above thinking and answers, choose the most consistent rearranged formula for ln(2)."
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3, final_instr3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3, "answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    # Stage 4
    debate_instr = "Sub-task 4: Compare the derived formula ln2=(T₁−T₂)/(T₁*T₂) to the four provided candidate equations and select the correct one. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents4 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N4 = self.max_round
    all_thinking4 = [[] for _ in range(N4)]
    all_answer4 = [[] for _ in range(N4)]
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":debate_instr,"context":["user query","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"Debate"}
    for r in range(N4):
        for i, agent in enumerate(debate_agents4):
            if r == 0:
                thinking4_i, answer4_i = await agent([taskInfo, thinking3, answer3], debate_instr, r, is_sub_task=True)
            else:
                thinking4_i, answer4_i = await agent([taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1], debate_instr, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
            all_thinking4[r].append(thinking4_i)
            all_answer4[r].append(answer4_i)
    final_decision_agent4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr4 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking4, answer4 = await final_decision_agent4([taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1], debate_instr + final_instr4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4, "answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs