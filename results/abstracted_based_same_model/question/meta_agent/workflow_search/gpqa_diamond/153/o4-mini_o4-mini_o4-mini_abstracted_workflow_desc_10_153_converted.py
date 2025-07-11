async def forward_153(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Parse the query to extract all relevant spectral data (mass spec peaks, IR peaks, 1H NMR data) and the list of structural choices."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_instruction,"context":["user query"],"agent_collaboration":"CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, parsing query, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"]={"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_instruction2 = "Sub-task 2: Parse the output instructions to confirm that the final answer must be returned as a single letter (A, B, C, or D)."
    cot_agent2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_instruction2,"context":["user query"],"agent_collaboration":"CoT"}
    thinking2, answer2 = await cot_agent2([taskInfo], cot_instruction2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2.id}, parsing instructions, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"]={"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction3 = "Sub-task 3: Interpret the mass spectrum: use the molecular ion peak at m/z 156 and the M+2 peak at m/z 158 (32%) to deduce the presence of a chlorine atom and estimate the molecular formula."
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_instruction3,"context":["user query","thinking1","answer1"],"agent_collaboration":"CoT"}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, interpreting mass spec, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"]={"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_instruction4 = "Sub-task 4: Interpret the IR data: identify functional groups responsible for the broad 3500–2700 cm⁻¹ absorption and the strong carbonyl peak at 1720 cm⁻¹."
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_instruction4,"context":["user query","thinking1","answer1"],"agent_collaboration":"CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking1, answer1], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, interpreting IR, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"]={"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_instruction5 = "Sub-task 5: Interpret the 1H NMR data: assign the signals at 11.0 ppm (s, 1H), 8.02 ppm (d, 2H), and 7.72 ppm (d, 2H) to specific protons and infer substitution pattern on the aromatic ring."
    cot_agent5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":cot_instruction5,"context":["user query","thinking1","answer1"],"agent_collaboration":"CoT"}
    thinking5, answer5 = await cot_agent5([taskInfo, thinking1, answer1], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, interpreting NMR, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"]={"thinking":thinking5,"answer":answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_sc_instruction6 = "Sub-task 6: Evaluate each candidate structure (choices A–D) against the interpreted spectroscopic features to score their compatibility."
    N = self.max_sc
    cot_agents6 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers6 = []
    thinkingmapping6 = {}
    answermapping6 = {}
    subtask_desc6 = {"subtask_id":"subtask_6","instruction":cot_sc_instruction6,"context":["user query","thinking3","answer3","thinking4","answer4","thinking5","answer5"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking_tmp, answer_tmp = await cot_agents6[i]([taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5], cot_sc_instruction6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents6[i].id}, scoring structures, thinking: {thinking_tmp.content}; answer: {answer_tmp.content}")
        possible_answers6.append(answer_tmp.content)
        thinkingmapping6[answer_tmp.content] = thinking_tmp
        answermapping6[answer_tmp.content] = answer_tmp
    answer6_content = Counter(possible_answers6).most_common(1)[0][0]
    thinking6 = thinkingmapping6[answer6_content]
    answer6 = answermapping6[answer6_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6["response"]={"thinking":thinking6,"answer":answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    debate_instruction7 = "Sub-task 7: Select the single letter (A, B, C, or D) corresponding to the structure that best matches all spectroscopic evidence, and format it according to the parsed instructions."
    debate_agents7 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max7)]
    all_answer7 = [[] for _ in range(N_max7)]
    subtask_desc7 = {"subtask_id":"subtask_7","instruction":debate_instruction7,"context":["user query","thinking6","answer6"],"agent_collaboration":"Debate"}
    for r in range(N_max7):
        for i, agent in enumerate(debate_agents7):
            if r==0:
                thinking_tmp, answer_tmp = await agent([taskInfo, thinking6, answer6], debate_instruction7, r, is_sub_task=True)
            else:
                input_infos7 = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking_tmp, answer_tmp = await agent(input_infos7, debate_instruction7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting final answer, thinking: {thinking_tmp.content}; answer: {answer_tmp.content}")
            all_thinking7[r].append(thinking_tmp)
            all_answer7[r].append(answer_tmp)
    final_agent7 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_agent7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the selected structure.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final structure, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7["response"]={"thinking":thinking7,"answer":answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs