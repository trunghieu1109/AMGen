async def forward_158(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1a = "Sub-task 1a: Determine whether the observed spectral feature at 790 nm is an emission line peak or a continuum/absorption break by evaluating the spectral shape and flux behavior around 790 nm, considering that flux drop shortward of 790 nm may indicate an absorption edge rather than an emission line."
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1a, answer1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, determining spectral feature type, thinking: {thinking1a.content}; answer: {answer1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc1a['response'] = {
        "thinking": thinking1a,
        "answer": answer1a
    }
    logs.append(subtask_desc1a)
    
    feature_type = answer1a.content.strip().lower()
    
    if "break" in feature_type or "absorption" in feature_type or "continuum" in feature_type:
        cot_instruction_1b = "Sub-task 1b: Since the feature at 790 nm is identified as a continuum or absorption break, consider plausible rest-frame spectral features causing flux drop shortward of 790 nm, specifically the Lyman-alpha forest cutoff at 121.6 nm and the Lyman limit at 91.2 nm, and list these as candidate rest wavelengths for redshift estimation."
        cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
        subtask_desc1b = {
            "subtask_id": "subtask_1b",
            "instruction": cot_instruction_1b,
            "context": ["user query", thinking1a, answer1a],
            "agent_collaboration": "CoT"
        }
        thinking1b, answer1b = await cot_agent_1b([taskInfo, thinking1a, answer1a], cot_instruction_1b, is_sub_task=True)
        agents.append(f"CoT agent {cot_agent_1b.id}, listing candidate rest wavelengths for break, thinking: {thinking1b.content}; answer: {answer1b.content}")
        sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
        subtask_desc1b['response'] = {
            "thinking": thinking1b,
            "answer": answer1b
        }
        logs.append(subtask_desc1b)
        candidate_rest_wavelengths_break = answer1b.content.strip()
        candidate_rest_wavelengths_emission = ""
    else:
        cot_instruction_1c = "Sub-task 1c: Since the feature at 790 nm is identified as an emission line peak, compile a list of common quasar emission lines in the near-infrared/optical range (e.g., H-alpha at 656.3 nm, H-beta at 486.1 nm, Mg II at 279.8 nm) to consider as candidate rest wavelengths for redshift estimation."
        cot_agent_1c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
        subtask_desc1c = {
            "subtask_id": "subtask_1c",
            "instruction": cot_instruction_1c,
            "context": ["user query", thinking1a, answer1a],
            "agent_collaboration": "CoT"
        }
        thinking1c, answer1c = await cot_agent_1c([taskInfo, thinking1a, answer1a], cot_instruction_1c, is_sub_task=True)
        agents.append(f"CoT agent {cot_agent_1c.id}, listing candidate rest wavelengths for emission, thinking: {thinking1c.content}; answer: {answer1c.content}")
        sub_tasks.append(f"Sub-task 1c output: thinking - {thinking1c.content}; answer - {answer1c.content}")
        subtask_desc1c['response'] = {
            "thinking": thinking1c,
            "answer": answer1c
        }
        logs.append(subtask_desc1c)
        candidate_rest_wavelengths_emission = answer1c.content.strip()
        candidate_rest_wavelengths_break = ""
    
    cot_sc_instruction_2 = "Sub-task 2: Calculate possible redshift values (z) for the quasar using the observed wavelength (790 nm) and each candidate rest-frame wavelength identified in subtasks 1b and 1c, clearly documenting the assumptions and resulting z values."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1a, answer1a],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        inputs_2 = [taskInfo, thinking1a, answer1a]
        if candidate_rest_wavelengths_break:
            inputs_2.append(candidate_rest_wavelengths_break)
        if candidate_rest_wavelengths_emission:
            inputs_2.append(candidate_rest_wavelengths_emission)
        thinking2, answer2 = await cot_agents_2[i](inputs_2, cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, calculating redshift candidates, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    
    cot_reflect_instruction_3 = "Sub-task 3: Perform a self-consistency check on the calculated redshift values by comparing each z with the observed spectral shape and flux drop characteristics to identify the most plausible redshift and corresponding spectral feature. Reflect on assumptions and uncertainties to finalize the redshift estimate."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1a, answer1a, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", thinking1a, answer1a, thinking2, answer2],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, performing self-consistency check on redshift, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the redshift self-consistency and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining redshift self-consistency, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    
    cot_sc_instruction_4 = "Sub-task 4: Using the Lambda-CDM cosmological parameters (H0=70 km/s/Mpc, Omega_m=0.3, Omega_Lambda=0.7, flat universe), compute the comoving distance corresponding to the most plausible redshift value determined in Sub-task 3. Use cosmological formulas and document assumptions."
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking3, answer3],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, computing comoving distance, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    
    debate_instruction_5 = "Sub-task 5: Compare the computed comoving distance from Sub-task 4 with the given choices (6, 7, 8, 9 Gpc) and select the closest value as the assumed comoving distance of the quasar from Earth. Debate the merits of each choice and finalize the selection."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", thinking4, answer4],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating comoving distance selection, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the closest comoving distance value.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing comoving distance selection, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs