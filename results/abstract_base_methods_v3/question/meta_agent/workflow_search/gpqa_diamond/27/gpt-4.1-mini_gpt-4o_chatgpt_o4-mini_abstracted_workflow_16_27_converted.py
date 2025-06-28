async def forward_27(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Analyze the starting material (S)-4-hydroxycyclohex-2-en-1-one by drawing its structure with explicit carbon numbering, identifying stereochemistry at C4, and noting reactive functional groups relevant for subsequent transformations."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing starting material, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    N = self.max_sc
    cot_sc_instruction_2 = "Sub-task 2: Determine the structure and stereochemical outcome of product 1 formed by protecting the hydroxy group of the starting material with tert-Butyldimethylsilyl chloride (TBDMSCl) and triethylamine, explicitly showing the protected site and retention of stereochemistry, based on Sub-task 1 outputs."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining product 1 structure and stereochemistry, thinking: {thinking2.content}; answer: {answer2.content}")
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
    print("Step 2: ", sub_tasks[-1])
    
    cot_sc_instruction_3a = "Sub-task 3a: Identify the nucleophile Ph2CuLi's correct mode of addition to product 1, confirming that it performs a 1,4-conjugate addition of a single phenyl group to the enone system, and determine the site and stereochemical outcome of this addition with explicit carbon numbering, based on Sub-task 2 outputs."
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
        agents.append(f"CoT-SC agent {cot_agents_3a[i].id}, analyzing Ph2CuLi 1,4-addition, thinking: {thinking3a.content}; answer: {answer3a.content}")
        possible_answers_3a.append(answer3a.content)
        thinkingmapping_3a[answer3a.content] = thinking3a
        answermapping_3a[answer3a.content] = answer3a
    answer3a_content = Counter(possible_answers_3a).most_common(1)[0][0]
    thinking3a = thinkingmapping_3a[answer3a_content]
    answer3a = answermapping_3a[answer3a_content]
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    
    cot_sc_instruction_3b = "Sub-task 3b: Analyze the subsequent alkylation of the enolate intermediate formed after Ph2CuLi addition by benzyl bromide, determining the site of benzylation and the stereochemical configuration of product 2, with detailed mechanistic rationale, based on Sub-task 3a outputs."
    cot_agents_3b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3b = []
    thinkingmapping_3b = {}
    answermapping_3b = {}
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_sc_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3b, answer3b = await cot_agents_3b[i]([taskInfo, thinking3a, answer3a], cot_sc_instruction_3b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3b[i].id}, analyzing benzyl bromide alkylation, thinking: {thinking3b.content}; answer: {answer3b.content}")
        possible_answers_3b.append(answer3b.content)
        thinkingmapping_3b[answer3b.content] = thinking3b
        answermapping_3b[answer3b.content] = answer3b
    answer3b_content = Counter(possible_answers_3b).most_common(1)[0][0]
    thinking3b = thinkingmapping_3b[answer3b_content]
    answer3b = answermapping_3b[answer3b_content]
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    
    cot_reflect_instruction_3r = "Sub-task 3 Reflexion: Perform a self-consistency check on the mechanistic reasoning and stereochemical assignments of product 2 from subtasks 3a and 3b, comparing alternative plausible pathways and selecting the most chemically sound structure."
    cot_agent_3r = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3r = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3r = self.max_round
    cot_inputs_3r = [taskInfo, thinking3a, answer3a, thinking3b, answer3b]
    subtask_desc3r = {
        "subtask_id": "subtask_3_reflexion",
        "instruction": cot_reflect_instruction_3r,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Reflexion"
    }
    thinking3r, answer3r = await cot_agent_3r(cot_inputs_3r, cot_reflect_instruction_3r, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3r.id}, reviewing product 2 mechanistic and stereochemical assignments, thinking: {thinking3r.content}; answer: {answer3r.content}")
    for i in range(N_max_3r):
        feedback, correct = await critic_agent_3r([taskInfo, thinking3r, answer3r], "please review the mechanistic and stereochemical reasoning of product 2 and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3r.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3r.extend([thinking3r, answer3r, feedback])
        thinking3r, answer3r = await cot_agent_3r(cot_inputs_3r, cot_reflect_instruction_3r, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3r.id}, refining product 2 assignments, thinking: {thinking3r.content}; answer: {answer3r.content}")
    sub_tasks.append(f"Sub-task 3 reflexion output: thinking - {thinking3r.content}; answer - {answer3r.content}")
    subtask_desc3r['response'] = {
        "thinking": thinking3r,
        "answer": answer3r
    }
    logs.append(subtask_desc3r)
    print("Step 3 reflexion: ", sub_tasks[-1])
    
    cot_sc_instruction_4a = "Sub-task 4a: List and number all possible enolizable α-positions in product 2, evaluating their relative acidity and steric hindrance to predict the preferred site of enolate formation upon treatment with LDA at low temperature, based on Sub-task 3 reflexion outputs."
    cot_agents_4a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4a = []
    thinkingmapping_4a = {}
    answermapping_4a = {}
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_sc_instruction_4a,
        "context": ["user query", "thinking of subtask 3 reflexion", "answer of subtask 3 reflexion"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4a, answer4a = await cot_agents_4a[i]([taskInfo, thinking3r, answer3r], cot_sc_instruction_4a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4a[i].id}, listing enolizable sites and evaluating acidity/sterics, thinking: {thinking4a.content}; answer: {answer4a.content}")
        possible_answers_4a.append(answer4a.content)
        thinkingmapping_4a[answer4a.content] = thinking4a
        answermapping_4a[answer4a.content] = answer4a
    answer4a_content = Counter(possible_answers_4a).most_common(1)[0][0]
    thinking4a = thinkingmapping_4a[answer4a_content]
    answer4a = answermapping_4a[answer4a_content]
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {
        "thinking": thinking4a,
        "answer": answer4a
    }
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])
    
    cot_sc_instruction_4b = "Sub-task 4b: Predict the stereochemical outcome of methylation at the preferred enolate site by iodomethane, considering kinetic vs thermodynamic control, conformational effects, and stereochemical induction, to determine the structure and stereochemistry of product 3, based on Sub-task 4a outputs."
    cot_agents_4b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4b = []
    thinkingmapping_4b = {}
    answermapping_4b = {}
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_sc_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4b, answer4b = await cot_agents_4b[i]([taskInfo, thinking4a, answer4a], cot_sc_instruction_4b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4b[i].id}, predicting methylation stereochemistry, thinking: {thinking4b.content}; answer: {answer4b.content}")
        possible_answers_4b.append(answer4b.content)
        thinkingmapping_4b[answer4b.content] = thinking4b
        answermapping_4b[answer4b.content] = answer4b
    answer4b_content = Counter(possible_answers_4b).most_common(1)[0][0]
    thinking4b = thinkingmapping_4b[answer4b_content]
    answer4b = answermapping_4b[answer4b_content]
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {
        "thinking": thinking4b,
        "answer": answer4b
    }
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])
    
    cot_reflect_instruction_4r = "Sub-task 4 Reflexion: Conduct a self-consistency and stereochemical validation of the predicted methylation site and stereochemical outcome in product 3, exploring alternative enolate formation and methylation scenarios to confirm the most plausible structure, based on Sub-tasks 4a and 4b outputs."
    cot_agent_4r = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4r = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4r = self.max_round
    cot_inputs_4r = [taskInfo, thinking4a, answer4a, thinking4b, answer4b]
    subtask_desc4r = {
        "subtask_id": "subtask_4_reflexion",
        "instruction": cot_reflect_instruction_4r,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "Reflexion"
    }
    thinking4r, answer4r = await cot_agent_4r(cot_inputs_4r, cot_reflect_instruction_4r, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4r.id}, validating methylation site and stereochemistry, thinking: {thinking4r.content}; answer: {answer4r.content}")
    for i in range(N_max_4r):
        feedback, correct = await critic_agent_4r([taskInfo, thinking4r, answer4r], "please review the methylation site and stereochemical outcome and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4r.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4r.extend([thinking4r, answer4r, feedback])
        thinking4r, answer4r = await cot_agent_4r(cot_inputs_4r, cot_reflect_instruction_4r, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4r.id}, refining methylation predictions, thinking: {thinking4r.content}; answer: {answer4r.content}")
    sub_tasks.append(f"Sub-task 4 reflexion output: thinking - {thinking4r.content}; answer - {answer4r.content}")
    subtask_desc4r['response'] = {
        "thinking": thinking4r,
        "answer": answer4r
    }
    logs.append(subtask_desc4r)
    print("Step 4 reflexion: ", sub_tasks[-1])
    
    cot_reflect_instruction_5 = "Sub-task 5: Determine the effect of aqueous HCl treatment on product 3, focusing on deprotection of the TBDMS group and any acid-mediated transformations, to deduce the structure and stereochemistry of the final product 4 with explicit carbon numbering, based on Sub-task 4 reflexion outputs."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking4r, answer4r]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", "thinking of subtask 4 reflexion", "answer of subtask 4 reflexion"],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, deducing final product 4 structure, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "please review the deprotection and acid-mediated transformations and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining final product 4 structure, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    debate_instruction_6 = "Sub-task 6: Compare the deduced structure and stereochemistry of final product 4 with the given multiple-choice options, analyzing stereochemical descriptors and substituent positions to identify the correct answer choice, based on Sub-task 5 outputs."
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing final product 4 with choices, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on the correct structure of product 4.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct product 4 structure, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    cot_reflect_instruction_7 = "Sub-task 7: Perform a final mechanistic sanity check on the overall reaction sequence and the assigned structure of product 4, verifying consistency with known reaction mechanisms, regioselectivity, and stereochemical outcomes to ensure correctness before final answer selection, based on Sub-task 6 outputs."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking6, answer6]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, performing final mechanistic sanity check, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max_7):
        feedback, correct = await critic_agent_7([taskInfo, thinking7, answer7], "please review the overall reaction mechanism and stereochemical assignments for correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining final sanity check, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs