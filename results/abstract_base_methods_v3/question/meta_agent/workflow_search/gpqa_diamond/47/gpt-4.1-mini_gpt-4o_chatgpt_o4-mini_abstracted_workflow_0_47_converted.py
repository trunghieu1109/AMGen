async def forward_47(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Identify the chemical structure of cyclohexanone, including ring size, functional groups, and relevant reactive sites, to establish the starting material for subsequent transformations. Provide a brief structural description and, if possible, a SMILES string or simple ASCII diagram."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identify cyclohexanone structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    N = self.max_sc
    cot_sc_instruction_2 = "Sub-task 2: Determine the structure of product 1 formed after treating cyclohexanone with bromine. Specify the site and type of bromination (alpha-bromination). Provide a structural representation (SMILES or ASCII) to clarify the product."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determine product 1 structure, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_sc_instruction_3a = "Sub-task 3a: Identify the reaction type occurring when product 1 is heated with sodium hydroxide. Recognize it as a Favorskii rearrangement and outline the general mechanistic features."
    cot_agents_3a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3a = []
    thinkingmapping_3a = {}
    answermapping_3a = {}
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3a, answer3a = await cot_agents_3a[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3a[i].id}, identify reaction type for product 1 heating, thinking: {thinking3a.content}; answer: {answer3a.content}")
        possible_answers_3a.append(answer3a.content)
        thinkingmapping_3a[answer3a.content] = thinking3a
        answermapping_3a[answer3a.content] = answer3a
    answer3a_content = Counter(possible_answers_3a).most_common(1)[0][0]
    thinking3a = thinkingmapping_3a[answer3a_content]
    answer3a = answermapping_3a[answer3a_content]
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    
    cot_reflect_instruction_3b = "Sub-task 3b: Explicitly analyze the Favorskii rearrangement mechanism for alpha-bromo cyclohexanone, emphasizing the ring contraction from a six-membered to a five-membered ring and formation of cyclopentanecarboxylic acid structure. Provide a clear mechanistic explanation and structural representation (SMILES or ASCII)."
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3b = self.max_round
    cot_inputs_3b = [taskInfo, thinking3a, answer3a]
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_reflect_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "Reflexion"
    }
    thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, analyze Favorskii rearrangement mechanism, thinking: {thinking3b.content}; answer: {answer3b.content}")
    for i in range(N_max_3b):
        feedback_3b, correct_3b = await critic_agent_3b([taskInfo, thinking3b, answer3b], "Please review the mechanistic explanation and ring contraction accuracy, and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3b.id}, providing feedback, thinking: {feedback_3b.content}; answer: {correct_3b.content}")
        if correct_3b.content == "True":
            break
        cot_inputs_3b.extend([thinking3b, answer3b, feedback_3b])
        thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, refining Favorskii rearrangement analysis, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    
    cot_reflect_instruction_3c = "Sub-task 3c: Verify and confirm the correct structure of product 2 after the Favorskii rearrangement, ensuring the ring contraction is accurately represented. Provide a clear structural depiction (SMILES or ASCII diagram) and confirm the product is cyclopentanecarboxylic acid, not cyclohexanecarboxylic acid."
    cot_agent_3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3c = self.max_round
    cot_inputs_3c = [taskInfo, thinking3b, answer3b]
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_reflect_instruction_3c,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Reflexion"
    }
    thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, cot_reflect_instruction_3c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, verify product 2 structure, thinking: {thinking3c.content}; answer: {answer3c.content}")
    for i in range(N_max_3c):
        feedback_3c, correct_3c = await critic_agent_3c([taskInfo, thinking3c, answer3c], "Please verify the ring size and product correctness, and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3c.id}, providing feedback, thinking: {feedback_3c.content}; answer: {correct_3c.content}")
        if correct_3c.content == "True":
            break
        cot_inputs_3c.extend([thinking3c, answer3c, feedback_3c])
        thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, cot_reflect_instruction_3c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, refining product 2 verification, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c['response'] = {"thinking": thinking3c, "answer": answer3c}
    logs.append(subtask_desc3c)
    print("Step 3c: ", sub_tasks[-1])
    
    cot_instruction_4 = "Sub-task 4: Determine the structure of product 3 formed after treating product 2 with thionyl chloride and pyridine. Identify the conversion of the carboxylic acid to the corresponding acid chloride and provide structural details (SMILES or ASCII)."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking3c, answer3c]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 3c", "answer of subtask 3c"],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, determine product 3 structure, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback_4, correct_4 = await critic_agent_4([taskInfo, thinking4, answer4], "Please review the functional group transformation and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback_4.content}; answer: {correct_4.content}")
        if correct_4.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback_4])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining product 3 structure, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    debate_instruction_5 = "Sub-task 5: Determine the structure of product 4 formed after reducing product 3 with Lithium tri-tert-butoxyaluminum hydride. Identify the selective reduction to the aldehyde functional group and provide stereochemical and structural details."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, determine product 4 structure, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on product 4 structure.", is_sub_task=True)
    agents.append(f"Final Decision agent, determining product 4 structure, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    cot_reflect_instruction_6 = "Sub-task 6: Analyze the final structure of product 4 to identify all chemically distinct hydrogen environments. Explicitly define chemically distinct hydrogens as unique hydrogen types differing in chemical environment or symmetry. Provide a detailed list of these environments with structural context."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, analyzing distinct hydrogen environments, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback_6, correct_6 = await critic_agent_6([taskInfo, thinking6, answer6], "Critically evaluate the identification of chemically distinct hydrogen environments and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback_6.content}; answer: {correct_6.content}")
        if correct_6.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback_6])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining distinct hydrogen environment analysis, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    cot_reflect_instruction_7a = "Sub-task 7a: Count the number of chemically distinct hydrogen environments identified in product 4 from Sub-task 6. Ensure the count reflects unique environments rather than total hydrogens. Map this count to the multiple-choice options provided in the query."
    cot_agent_7a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    subtask_desc7a = {
        "subtask_id": "subtask_7a",
        "instruction": cot_reflect_instruction_7a,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "CoT"
    }
    thinking7a, answer7a = await cot_agent_7a([taskInfo, thinking6, answer6], cot_reflect_instruction_7a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7a.id}, count distinct hydrogen environments, thinking: {thinking7a.content}; answer: {answer7a.content}")
    sub_tasks.append(f"Sub-task 7a output: thinking - {thinking7a.content}; answer - {answer7a.content}")
    subtask_desc7a['response'] = {"thinking": thinking7a, "answer": answer7a}
    logs.append(subtask_desc7a)
    print("Step 7a: ", sub_tasks[-1])
    
    cot_reflect_instruction_7b = "Sub-task 7b: Perform a reflexion and self-consistency check on the counting of chemically distinct hydrogens from Sub-task 7a. Verify alignment with chemical definitions and the question's intent. Correct any discrepancies before finalizing the answer."
    cot_agent_7b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7b = self.max_round
    cot_inputs_7b = [taskInfo, thinking7a, answer7a]
    subtask_desc7b = {
        "subtask_id": "subtask_7b",
        "instruction": cot_reflect_instruction_7b,
        "context": ["user query", "thinking of subtask 7a", "answer of subtask 7a"],
        "agent_collaboration": "Reflexion"
    }
    thinking7b, answer7b = await cot_agent_7b(cot_inputs_7b, cot_reflect_instruction_7b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7b.id}, verify counting of distinct hydrogens, thinking: {thinking7b.content}; answer: {answer7b.content}")
    for i in range(N_max_7b):
        feedback_7b, correct_7b = await critic_agent_7b([taskInfo, thinking7b, answer7b], "Critically evaluate the counting and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7b.id}, providing feedback, thinking: {feedback_7b.content}; answer: {correct_7b.content}")
        if correct_7b.content == "True":
            break
        cot_inputs_7b.extend([thinking7b, answer7b, feedback_7b])
        thinking7b, answer7b = await cot_agent_7b(cot_inputs_7b, cot_reflect_instruction_7b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7b.id}, refining counting verification, thinking: {thinking7b.content}; answer: {answer7b.content}")
    sub_tasks.append(f"Sub-task 7b output: thinking - {thinking7b.content}; answer - {answer7b.content}")
    subtask_desc7b['response'] = {"thinking": thinking7b, "answer": answer7b}
    logs.append(subtask_desc7b)
    print("Step 7b: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking7b, answer7b, sub_tasks, agents)
    return final_answer, logs