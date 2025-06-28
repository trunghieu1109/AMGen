async def forward_45(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Identify and characterize the molecular structure of racemic 3-methylpent-1-ene, including the position of the double bond, the chiral center(s), and the stereochemical configuration of the starting material, to establish a precise structural basis for the reaction analysis."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying and characterizing racemic 3-methylpent-1-ene, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = (
        "Sub-task 2: Describe the detailed mechanism of Grubbs catalyst-mediated self-metathesis of terminal alkenes, "
        "explicitly illustrating the formation of the dimeric product with a single internal double bond and ethene as the co-product; "
        "provide structural representations (e.g., SMILES or ASCII diagrams) of both the starting alkene and the expected dimeric product to ensure clarity and accuracy, "
        "based on the structure characterized in Sub-task 1."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, describing Grubbs catalyst self-metathesis mechanism, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_reflect_instruction_3a = (
        "Sub-task 3a: Generate the precise chemical structure of the dimeric product formed by self-metathesis of racemic 3-methylpent-1-ene, "
        "confirming the number and position of double bonds and the connectivity of the carbon skeleton, based on the mechanism described in Sub-task 2."
    )
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_reflect_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking2, answer2], cot_reflect_instruction_3a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, generating dimeric product structure, thinking: {thinking3a.content}; answer: {answer3a.content}")
    critic_agent_3a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3a = self.max_round
    cot_inputs_3a = [taskInfo, thinking2, answer2]
    for i in range(N_max_3a):
        feedback, correct = await critic_agent_3a([taskInfo, thinking3a, answer3a], "Please review the dimeric product structure and provide any limitations or corrections.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3a.id}, providing feedback on dimeric product structure, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3a.extend([thinking3a, answer3a, feedback])
        thinking3a, answer3a = await cot_agent_3a(cot_inputs_3a, cot_reflect_instruction_3a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, refining dimeric product structure, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    
    cot_reflect_instruction_3b = (
        "Sub-task 3b: Enumerate all possible stereoisomers of the dimeric product arising from E/Z (cis/trans) isomerism of the single internal double bond, "
        "considering the stereochemical constraints imposed by the product structure generated in Sub-task 3a."
    )
    debate_agents_3b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3b = self.max_round
    all_thinking3b = [[] for _ in range(N_max_3b)]
    all_answer3b = [[] for _ in range(N_max_3b)]
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_reflect_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3b):
        for i, agent in enumerate(debate_agents_3b):
            if r == 0:
                thinking3b, answer3b = await agent([taskInfo, thinking3a, answer3a], cot_reflect_instruction_3b, r, is_sub_task=True)
            else:
                input_infos_3b = [taskInfo, thinking3a, answer3a] + all_thinking3b[r-1] + all_answer3b[r-1]
                thinking3b, answer3b = await agent(input_infos_3b, cot_reflect_instruction_3b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, enumerating E/Z stereoisomers, thinking: {thinking3b.content}; answer: {answer3b.content}")
            all_thinking3b[r].append(thinking3b)
            all_answer3b[r].append(answer3b)
    final_decision_agent_3b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3b, answer3b = await final_decision_agent_3b([taskInfo] + all_thinking3b[-1] + all_answer3b[-1], "Sub-task 3b: Make a final decision on the number of E/Z stereoisomers.", is_sub_task=True)
    agents.append(f"Final Decision agent on E/Z stereoisomers, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    
    cot_reflect_instruction_3c = (
        "Sub-task 3c: Analyze the stereochemical configurations at the chiral centers of the dimeric product, "
        "explicitly identifying enantiomeric pairs, meso forms, and symmetry elements to eliminate redundant or equivalent stereoisomers, "
        "thereby refining the count of unique stereoisomers, based on the enumeration from Sub-task 3b."
    )
    debate_agents_3c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3c = self.max_round
    all_thinking3c = [[] for _ in range(N_max_3c)]
    all_answer3c = [[] for _ in range(N_max_3c)]
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_reflect_instruction_3c,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3c):
        for i, agent in enumerate(debate_agents_3c):
            if r == 0:
                thinking3c, answer3c = await agent([taskInfo, thinking3b, answer3b], cot_reflect_instruction_3c, r, is_sub_task=True)
            else:
                input_infos_3c = [taskInfo, thinking3b, answer3b] + all_thinking3c[r-1] + all_answer3c[r-1]
                thinking3c, answer3c = await agent(input_infos_3c, cot_reflect_instruction_3c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing chiral stereochemistry and symmetry, thinking: {thinking3c.content}; answer: {answer3c.content}")
            all_thinking3c[r].append(thinking3c)
            all_answer3c[r].append(answer3c)
    final_decision_agent_3c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3c, answer3c = await final_decision_agent_3c([taskInfo] + all_thinking3c[-1] + all_answer3c[-1], "Sub-task 3c: Make a final decision on the refined count of unique stereoisomers considering meso forms and symmetry.", is_sub_task=True)
    agents.append(f"Final Decision agent on refined stereoisomer count, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c['response'] = {
        "thinking": thinking3c,
        "answer": answer3c
    }
    logs.append(subtask_desc3c)
    print("Step 3c: ", sub_tasks[-1])
    
    debate_instruction_4 = (
        "Sub-task 4: Critically review and consolidate the stereoisomer enumeration results from previous subtasks, "
        "perform sanity checks against stereochemical principles (including the 2^n rule adjusted for meso forms and symmetry), "
        "and provide a final, justified count of the distinct possible products formed (excluding ethene) from the Grubbs catalyst-mediated self-metathesis of racemic 3-methylpent-1-ene."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3c", "answer of subtask 3c"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3c, answer3c], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3c, answer3c] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, consolidating stereoisomer count and sanity checks, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make a final decision on the number of distinct possible products excluding ethene, with justification.", is_sub_task=True)
    agents.append(f"Final Decision agent on final product count, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs