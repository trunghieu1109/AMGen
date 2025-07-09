async def forward_152(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Extract and list, for each reaction (A, B, C), all named reactants, reagents, solvents, workup conditions, and define any unlabeled placeholders (e.g., define C explicitly as cyclohexane-1,3-dione)."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_instruction,"context":["user query"],"agent_collaboration":"CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting reaction components, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"]={"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_instruction2 = "Sub-task 2: Validate the completeness and correctness of the extracted reaction components by running a checklist: each reaction must have a clearly named nucleophile (enolate precursor), α-β unsaturated acceptor, base, solvent, and workup. Confirm placeholder C is defined unambiguously."
    cot_agent2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_instruction2,"context":["user query","thinking of subtask_1","answer of subtask_1"],"agent_collaboration":"CoT"}
    thinking2, answer2 = await cot_agent2([taskInfo,thinking1,answer1], cot_instruction2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2.id}, validating components, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"]={"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction3 = "Sub-task 3: Classify each reaction as a Michael addition: identify which component forms the enolate nucleophile, which is the α-β unsaturated Michael acceptor, and describe the key C–C bond formation at the β-carbon."
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_instruction3,"context":["user query","thinking of subtask_1","answer of subtask_1","thinking of subtask_2","answer of subtask_2"],"agent_collaboration":"CoT"}
    thinking3, answer3 = await cot_agent3([taskInfo,thinking1,answer1,thinking2,answer2], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, classifying Michael additions, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"]={"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    sc_instruction4 = "Sub-task 4: Generate skeletal outlines of the Michael addition products A, B, and C: produce at least three independent sketches or text outlines for each product and cross-compare to converge on the correct skeleton."
    N = self.max_sc
    sc_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_outlines = []
    thinking_map4 = {}
    answer_map4 = {}
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":sc_instruction4,"context":["user query","thinking of subtask_3","answer of subtask_3"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking4_i, answer4_i = await sc_agents[i]([taskInfo,thinking3,answer3], sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents[i].id}, generating skeleton outline, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_outlines.append(answer4_i.content)
        thinking_map4[answer4_i.content] = thinking4_i
        answer_map4[answer4_i.content] = answer4_i
    chosen4 = Counter(possible_outlines).most_common(1)[0][0]
    thinking4 = thinking_map4[chosen4]
    answer4 = answer_map4[chosen4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"]={"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    reflect_instruction5 = "Sub-task 5: Reflexively review and compare the converged skeletons against reaction stoichiometry, side-chains, and possible tautomeric forms; confirm correct placement of esters, nitriles, and keto/enol tautomers before naming."
    reflect_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    inputs5 = [taskInfo,thinking1,answer1,thinking2,answer2,thinking3,answer3,thinking4,answer4]
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":reflect_instruction5,"context":["user query","response of subtask_4"],"agent_collaboration":"Reflexion"}
    thinking5, answer5 = await reflect_agent(inputs5, reflect_instruction5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {reflect_agent.id}, reviewing skeletons, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max):
        feedback5, correct5 = await critic_agent([taskInfo,thinking5,answer5], "Review the skeleton review step and note any missing or incorrect placements.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback5.content}; correct: {correct5.content}")
        if correct5.content == "True":
            break
        inputs5.extend([thinking5,answer5,feedback5])
        thinking5, answer5 = await reflect_agent(inputs5, reflect_instruction5, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {reflect_agent.id}, refining review, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"]={"thinking":thinking5,"answer":answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    debate_instruction6 = "Sub-task 6: Install full substituents on each verified skeleton and derive systematic IUPAC names for products A, B, and C, including stereochemical descriptors if needed."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    rounds = self.max_round
    all_thinking6 = [[] for _ in range(rounds)]
    all_answer6 = [[] for _ in range(rounds)]
    subtask_desc6 = {"subtask_id":"subtask_6","instruction":debate_instruction6,"context":["user query","response of subtask_5"],"agent_collaboration":"Debate"}
    for r in range(rounds):
        for idx, agent in enumerate(debate_agents):
            if r == 0:
                thinking6_i, answer6_i = await agent([taskInfo,thinking5,answer5], debate_instruction6, r, is_sub_task=True)
            else:
                prev = all_thinking6[r-1] + all_answer6[r-1]
                thinking6_i, answer6_i = await agent([taskInfo,thinking5,answer5] + prev, debate_instruction6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking6_i.content}; answer: {answer6_i.content}")
            all_thinking6[r].append(thinking6_i)
            all_answer6[r].append(answer6_i)
    final_decision6 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on IUPAC names for A, B, and C.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision6.id}, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6["response"]={"thinking":thinking6,"answer":answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    cot_instruction7 = "Sub-task 7: Perform a final consistency check: compare the derived IUPAC names of A, B, and C to the four provided multiple-choice options and select the single matching choice (or flag if none match)."
    cot_agent7 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {"subtask_id":"subtask_7","instruction":cot_instruction7,"context":["user query","thinking of subtask_6","answer of subtask_6"],"agent_collaboration":"CoT"}
    thinking7, answer7 = await cot_agent7([taskInfo,thinking6,answer6], cot_instruction7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent7.id}, selecting matching choice, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7["response"]={"thinking":thinking7,"answer":answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs