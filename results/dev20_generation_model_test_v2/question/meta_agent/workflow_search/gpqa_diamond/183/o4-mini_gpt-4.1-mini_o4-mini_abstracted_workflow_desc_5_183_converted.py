async def forward_183(self, taskInfo):
    from collections import Counter
    print("Task Requirement:", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    # Stage 1: Extract and summarize given information (SC_CoT)
    cot_sc_instruction = "Sub-task 1: Extract and summarize the starting material, target substitution pattern, available reagents, and clarify numbering conventions, avoiding assumptions until the first substituent is placed."
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N1):
        thinking, answer = await cot_agents1[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings1.append(thinking)
        possible_answers1.append(answer)
    final_decision1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision1([taskInfo] + possible_thinkings1 + possible_answers1, "Sub-task 1: Synthesize the most consistent summary." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1, "answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Stage 2: Classify reagents and directing effects (SC_CoT)
    cot_sc_instruction2 = "Sub-task 2: Classify each reagent step by its functional role and state the directing effects of the installed group on future electrophilic substitutions."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction2,"context":["user query","thinking1","answer1"],"agent_collaboration":"SC_CoT"}
    for i in range(N2):
        thinking, answer = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings2.append(thinking)
        possible_answers2.append(answer)
    final_decision2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Sub-task 2: Synthesize the most consistent classification." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2, "answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Stage 3: Propose sequences via Debate
    debate_instr3 = "Sub-task 3: Propose plausible sequences ordering the nine steps to install tert-butyl, ethoxy, and nitro in the correct positions, justifying each step. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents3 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = [[] for _ in range(self.max_round)]
    all_answer3 = [[] for _ in range(self.max_round)]
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":debate_instr3,"context":["user query","thinking2","answer2"],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for agent in debate_agents3:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking2, answer2], debate_instr3, r, is_sub_task=True)
            else:
                thinking, answer = await agent([taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1], debate_instr3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking3[r].append(thinking)
            all_answer3[r].append(answer)
    final_decision3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr3 = "Given all the above thinking and answers, reason over them carefully and provide a final sequence proposal."
    thinking3, answer3 = await final_decision3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3:" + final_instr3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3, "answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    # Stage 4: Evaluate sequences via Debate
    debate_instr4 = "Sub-task 4: Evaluate each proposed sequence on regiochemical fidelity, step economy, and yield potential. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated evaluation."
    debate_agents4 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(self.max_round)]
    all_answer4 = [[] for _ in range(self.max_round)]
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":debate_instr4,"context":["user query","thinking3","answer3"],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for agent in debate_agents4:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking3, answer3], debate_instr4, r, is_sub_task=True)
            else:
                thinking, answer = await agent([taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1], debate_instr4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking4[r].append(thinking)
            all_answer4[r].append(answer)
    final_decision4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr4 = "Given all the above thinking and answers, reason over them carefully and provide a final evaluation."
    thinking4, answer4 = await final_decision4([taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4:" + final_instr4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4, "answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs