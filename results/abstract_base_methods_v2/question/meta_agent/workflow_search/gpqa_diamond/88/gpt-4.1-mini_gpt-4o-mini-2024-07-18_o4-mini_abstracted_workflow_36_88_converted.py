async def forward_88(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_agents_sc = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_reflexion = self.max_round
    N_max_debate = self.max_round
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)

    cot_instruction_1 = (
        "Sub-task 1: Propose all plausible reaction mechanisms and product structures for the reaction of 1,3-dibromoadamantane heated with excess KOH at 240°C, "
        "including the Favorskii rearrangement pathway. Provide candidate structures for product 1 with explicit atom connectivity (e.g., SMILES) and consider stereochemistry where relevant."
    )
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}
    for i in range(self.max_sc):
        thinking1, answer1 = await cot_agents_sc[i]([taskInfo], cot_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, proposing mechanisms and structures for product 1, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1.content)
        thinkingmapping_1[answer1.content] = thinking1
        answermapping_1[answer1.content] = answer1
    answer1_content = Counter(possible_answers_1).most_common(1)[0][0]
    thinking1 = thinkingmapping_1[answer1_content]
    answer1 = answermapping_1[answer1_content]
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Perform detailed spectral interpretation of product 1 using the provided 1H NMR and IR data: analyze chemical shifts, integration, multiplicities, "
        "and the IR absorbance at 1720 cm⁻¹ to confirm or refute the proposed structures from Subtask 1. Quantitatively correlate spectral features with structural elements."
    )
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_agents_sc[i]([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, interpreting spectral data for product 1, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Iteratively verify and select the most consistent structure for product 1 by cross-validating proposed mechanisms and spectral data. "
        "Output a detailed JSON including the confirmed structure, spectral match scores, and conclusion."
    )
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, verifying product 1 structure, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_reflexion):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "Please review the verification of product 1 structure and spectral consistency and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining verification of product 1, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Propose the chemical transformation and plausible structure(s) of product 2 formed by heating product 1 with excess aluminum isopropoxide, "
        "incorporating the Oppenauer oxidation mechanism. Provide explicit atom connectivity and predict key spectral features expected for product 2."
    )
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    for i in range(self.max_sc):
        thinking4, answer4 = await cot_agents_sc[i]([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, proposing product 2 structure and transformation, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = (
        "Sub-task 5: Verify the proposed structure of product 2 by comparing predicted spectral features with known data or literature, "
        "ensuring consistency with the reaction conditions and product 1 structure. Provide a detailed rationale and output a JSON with structure and spectral validation."
    )
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, verifying product 2 structure, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_reflexion):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "Please review the verification of product 2 structure and spectral consistency and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining verification of product 2, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6 = (
        "Sub-task 6: Propose the structure of product 3 formed by ozonolysis of product 2 at -78°C followed by dimethyl sulfide treatment, "
        "detailing the reaction mechanism and expected functional groups. Provide explicit atom connectivity and predict key 1H NMR features relevant to the most deshielded hydrogen."
    )
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_6 = []
    thinkingmapping_6 = {}
    answermapping_6 = {}
    for i in range(self.max_sc):
        thinking6, answer6 = await cot_agents_sc[i]([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, proposing product 3 structure and mechanism, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6.content)
        thinkingmapping_6[answer6.content] = thinking6
        answermapping_6[answer6.content] = answer6
    answer6_content = Counter(possible_answers_6).most_common(1)[0][0]
    thinking6 = thinkingmapping_6[answer6_content]
    answer6 = answermapping_6[answer6_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6["response"] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_instruction_7 = (
        "Sub-task 7: Validate the proposed structure of product 3 by cross-checking predicted spectral data with the experimental 1H NMR spectrum, "
        "focusing on chemical shifts, integration, and multiplicities. Output a detailed JSON with structure, spectral correlation, and confidence level."
    )
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_7 = [taskInfo, thinking6, answer6]
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, validating product 3 structure, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max_reflexion):
        feedback, correct = await critic_agent_7([taskInfo, thinking7, answer7], "Please review the validation of product 3 structure and spectral data and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining validation of product 3, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7["response"] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    cot_instruction_8 = (
        "Sub-task 8: Identify the most deshielded hydrogen atom in product 3's 1H NMR spectrum (excluding exchangeable hydrogens) based on the confirmed structure and spectral data. "
        "Provide atom labeling and environment description."
    )
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7"],
        "agent_collaboration": "CoT"
    }
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, identifying most deshielded hydrogen in product 3, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8["response"] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    cot_instruction_9 = (
        "Sub-task 9: Analyze the coupling pattern (multiplicity) of the most deshielded hydrogen atom in product 3 by counting neighboring hydrogens, considering coupling constants, "
        "and predicting splitting patterns. Provide detailed reasoning and expected multiplicity."
    )
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_instruction_9,
        "context": ["user query", "thinking of subtask 8", "answer of subtask 8"],
        "agent_collaboration": "CoT"
    }
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await cot_agent_9([taskInfo, thinking8, answer8], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, analyzing coupling pattern of most deshielded hydrogen, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9["response"] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    debate_instruction_10 = (
        "Sub-task 10: Match the predicted coupling pattern of the most deshielded hydrogen atom to the given multiple-choice options: doublet of triplets, triplet of triplets, pentet, triplet. "
        "Select the correct answer with justification."
    )
    subtask_desc10 = {
        "subtask_id": "subtask_10",
        "instruction": debate_instruction_10,
        "context": ["user query", "thinking of subtask 9", "answer of subtask 9"],
        "agent_collaboration": "Debate"
    }
    all_thinking10 = [[] for _ in range(N_max_debate)]
    all_answer10 = [[] for _ in range(N_max_debate)]
    for r in range(N_max_debate):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking10, answer10 = await agent([taskInfo, thinking9, answer9], debate_instruction_10, r, is_sub_task=True)
            else:
                input_infos_10 = [taskInfo, thinking9, answer9] + all_thinking10[r-1] + all_answer10[r-1]
                thinking10, answer10 = await agent(input_infos_10, debate_instruction_10, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, matching coupling pattern, thinking: {thinking10.content}; answer: {answer10.content}")
            all_thinking10[r].append(thinking10)
            all_answer10[r].append(answer10)
    thinking10, answer10 = await final_decision_agent([taskInfo] + all_thinking10[-1] + all_answer10[-1], "Sub-task 10: Make final decision on the coupling pattern of the most deshielded hydrogen atom in product 3's 1H NMR spectrum.", is_sub_task=True)
    agents.append(f"Final Decision agent, determining final coupling pattern, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10["response"] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])

    cot_instruction_11 = (
        "Sub-task 11: Perform a final consistency audit by re-evaluating the entire reasoning chain from product 1 through product 3, "
        "ensuring all structural assignments and spectral interpretations are coherent and consistent with the original data before finalizing the answer."
    )
    subtask_desc11 = {
        "subtask_id": "subtask_11",
        "instruction": cot_instruction_11,
        "context": ["user query"] + [thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5, thinking6, answer6, thinking7, answer7, thinking8, answer8, thinking9, answer9, thinking10, answer10],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_11 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_11 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_11 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5, thinking6, answer6, thinking7, answer7, thinking8, answer8, thinking9, answer9, thinking10, answer10]
    thinking11, answer11 = await cot_agent_11(cot_inputs_11, cot_instruction_11, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_11.id}, performing final consistency audit, thinking: {thinking11.content}; answer: {answer11.content}")
    for i in range(N_max_reflexion):
        feedback, correct = await critic_agent_11([taskInfo, thinking11, answer11], "Please review the entire reasoning chain for consistency and correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_11.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_11.extend([thinking11, answer11, feedback])
        thinking11, answer11 = await cot_agent_11(cot_inputs_11, cot_instruction_11, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_11.id}, refining final consistency audit, thinking: {thinking11.content}; answer: {answer11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}")
    subtask_desc11["response"] = {"thinking": thinking11, "answer": answer11}
    logs.append(subtask_desc11)
    print("Step 11: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking11, answer11, sub_tasks, agents)
    return final_answer, logs
