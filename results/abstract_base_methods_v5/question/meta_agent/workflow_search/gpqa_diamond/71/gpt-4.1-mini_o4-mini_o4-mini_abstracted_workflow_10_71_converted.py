async def forward_71(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Sub-task 1: Interpret the chemical structure and stereochemistry of the starting material 7-(tert-butoxy)bicyclo[2.2.1]hepta-2,5-diene, identifying reactive sites and structural features relevant for subsequent reactions."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, interpreting starting material structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 2: Analyze the reaction of 7-(tert-butoxy)bicyclo[2.2.1]hepta-2,5-diene with 2 equivalents of 5,6-bis(dibromomethyl)cyclohexa-1,3-diene and sodium iodide at elevated temperature, deducing the detailed structure and stereochemistry of product 1 formed, including mechanistic rationale."
    N2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing reaction to form product 1, thinking: {thinking2.content}; answer: {answer2.content}")
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

    cot_reflect_instruction_3 = "Sub-task 3: Determine the structural and stereochemical changes when product 1 is treated with aqueous sulfuric acid to form product 2, including hydrolysis, rearrangements, or functional group transformations, supported by mechanistic reasoning."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking and answer of subtask 1", "thinking and answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, determining changes to form product 2, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the structural changes and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining structural changes, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4a = "Sub-task 4a: Compile reagent profiles and typical reaction pathways for SO3 and pyridine in DMSO, including sulfonation, oxidation, and Ramberg–Bäcklund elimination, to inform mechanistic hypotheses."
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_instruction_4a,
        "context": ["user query", "thinking and answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking4a, answer4a = await cot_agent_4a([taskInfo, thinking3, answer3], cot_instruction_4a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4a.id}, compiling reagent profiles and typical pathways, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {"thinking": thinking4a, "answer": answer4a}
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])

    cot_instruction_4b = "Sub-task 4b: Enumerate plausible mechanistic routes for the reaction of product 2 with SO3 and pyridine in DMSO, predicting intermediate structures and transformations."
    N4b = self.max_sc
    cot_agents_4b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4b)]
    possible_answers_4b = []
    thinkingmapping_4b = {}
    answermapping_4b = {}
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_instruction_4b,
        "context": ["user query", "thinking and answer of subtask 3", "thinking and answer of subtask 4a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N4b):
        thinking4b, answer4b = await cot_agents_4b[i]([taskInfo, thinking3, answer3, thinking4a, answer4a], cot_instruction_4b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4b[i].id}, enumerating mechanistic routes for product 3, thinking: {thinking4b.content}; answer: {answer4b.content}")
        possible_answers_4b.append(answer4b.content)
        thinkingmapping_4b[answer4b.content] = thinking4b
        answermapping_4b[answer4b.content] = answer4b
    answer4b_content = Counter(possible_answers_4b).most_common(1)[0][0]
    thinking4b = thinkingmapping_4b[answer4b_content]
    answer4b = answermapping_4b[answer4b_content]
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {"thinking": thinking4b, "answer": answer4b}
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])

    cot_reflect_instruction_4c = "Sub-task 4c: Validate the proposed intermediate structures and mechanistic pathways for product 3 against known chemistry and reagent-function databases, selecting the most credible product 3 structure and assigning a confidence level."
    cot_agent_4c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4c = self.max_round
    cot_inputs_4c = [taskInfo, thinking3, answer3, thinking4a, answer4a, thinking4b, answer4b]
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": cot_reflect_instruction_4c,
        "context": ["user query", "thinking and answer of subtasks 3, 4a, 4b"],
        "agent_collaboration": "Reflexion"
    }
    thinking4c, answer4c = await cot_agent_4c(cot_inputs_4c, cot_reflect_instruction_4c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4c.id}, validating mechanistic pathways and selecting product 3 structure, thinking: {thinking4c.content}; answer: {answer4c.content}")
    for i in range(N_max_4c):
        feedback, correct = await critic_agent_4c([taskInfo, thinking4c, answer4c], "please review the mechanistic validation and confidence assignment.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4c.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4c.extend([thinking4c, answer4c, feedback])
        thinking4c, answer4c = await cot_agent_4c(cot_inputs_4c, cot_reflect_instruction_4c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4c.id}, refining mechanistic validation, thinking: {thinking4c.content}; answer: {answer4c.content}")
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking4c.content}; answer - {answer4c.content}")
    subtask_desc4c['response'] = {"thinking": thinking4c, "answer": answer4c}
    logs.append(subtask_desc4c)
    print("Step 4c: ", sub_tasks[-1])

    debate_instruction_5 = "Sub-task 5: Analyze the thermal reaction of product 3 heated at 150°C to form the final product 4 by: (a) proposing possible reaction pathways (e.g., cyclization, elimination, rearrangement), (b) predicting the resulting structure and stereochemistry of product 4, and (c) validating the predicted structure for chemical plausibility and consistency with prior steps."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking and answer of subtask 4c"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4c, answer4c], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4c, answer4c] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing thermal reaction to form product 4, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the structural changes and stereochemistry of final product 4.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding on final product 4 structure and stereochemistry, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6 = "Sub-task 6: Construct or obtain a detailed, stereochemically accurate chemical structure of the final product 4, incorporating all transformations and validated mechanistic insights from previous subtasks, to enable precise analysis of hydrogen environments."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking and answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, constructing final product 4 structure, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_instruction_7a = "Sub-task 7a: Analyze molecular symmetry and stereochemistry of final product 4 to identify chemically distinct hydrogen environments, considering electronic and steric factors."
    cot_agent_7a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7a = {
        "subtask_id": "subtask_7a",
        "instruction": cot_instruction_7a,
        "context": ["user query", "thinking and answer of subtask 6"],
        "agent_collaboration": "CoT"
    }
    thinking7a, answer7a = await cot_agent_7a([taskInfo, thinking6, answer6], cot_instruction_7a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7a.id}, analyzing symmetry and stereochemistry for hydrogen classification, thinking: {thinking7a.content}; answer: {answer7a.content}")
    sub_tasks.append(f"Sub-task 7a output: thinking - {thinking7a.content}; answer - {answer7a.content}")
    subtask_desc7a['response'] = {"thinking": thinking7a, "answer": answer7a}
    logs.append(subtask_desc7a)
    print("Step 7a: ", sub_tasks[-1])

    cot_instruction_7b = "Sub-task 7b: Use molecular visualization or computational chemistry tools to confirm hydrogen equivalence and distinctness in final product 4, validating stereochemical assignments."
    cot_agent_7b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7b = {
        "subtask_id": "subtask_7b",
        "instruction": cot_instruction_7b,
        "context": ["user query", "thinking and answer of subtask 7a"],
        "agent_collaboration": "CoT"
    }
    thinking7b, answer7b = await cot_agent_7b([taskInfo, thinking7a, answer7a], cot_instruction_7b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7b.id}, confirming hydrogen equivalence with computational tools, thinking: {thinking7b.content}; answer: {answer7b.content}")
    sub_tasks.append(f"Sub-task 7b output: thinking - {thinking7b.content}; answer - {answer7b.content}")
    subtask_desc7b['response'] = {"thinking": thinking7b, "answer": answer7b}
    logs.append(subtask_desc7b)
    print("Step 7b: ", sub_tasks[-1])

    cot_sc_instruction_8 = "Sub-task 8: Count the number of chemically distinct hydrogen atoms identified in product 4 and compare this count with the provided multiple-choice options (7, 8, 10, 4) to select the correct answer, based on Sub-tasks 7a and 7b outputs."
    N8 = self.max_sc
    cot_agents_8 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N8)]
    possible_answers_8 = []
    thinkingmapping_8 = {}
    answermapping_8 = {}
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_sc_instruction_8,
        "context": ["user query", "thinking and answer of subtask 7b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N8):
        thinking8, answer8 = await cot_agents_8[i]([taskInfo, thinking7b, answer7b], cot_sc_instruction_8, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_8[i].id}, counting distinct hydrogens and selecting answer, thinking: {thinking8.content}; answer: {answer8.content}")
        possible_answers_8.append(answer8.content)
        thinkingmapping_8[answer8.content] = thinking8
        answermapping_8[answer8.content] = answer8
    answer8_content = Counter(possible_answers_8).most_common(1)[0][0]
    thinking8 = thinkingmapping_8[answer8_content]
    answer8 = answermapping_8[answer8_content]
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    cot_reflect_instruction_9 = "Sub-task 9: Perform an independent verification and reflexion step to cross-check the predicted hydrogen count and final structure against literature precedents or experimental data for similar compounds, ensuring confidence in the final answer."
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_9 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_9 = self.max_round
    cot_inputs_9 = [taskInfo, thinking8, answer8]
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_reflect_instruction_9,
        "context": ["user query", "thinking and answer of subtask 8"],
        "agent_collaboration": "Reflexion"
    }
    thinking9, answer9 = await cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_9.id}, verifying hydrogen count and final structure, thinking: {thinking9.content}; answer: {answer9.content}")
    for i in range(N_max_9):
        feedback, correct = await critic_agent_9([taskInfo, thinking9, answer9], "please review the final hydrogen count and structural verification.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_9.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_9.extend([thinking9, answer9, feedback])
        thinking9, answer9 = await cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_9.id}, refining verification, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs
