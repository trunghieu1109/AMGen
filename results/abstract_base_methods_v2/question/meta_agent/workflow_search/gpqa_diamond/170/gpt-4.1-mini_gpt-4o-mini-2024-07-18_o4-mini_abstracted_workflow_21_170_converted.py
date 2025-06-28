async def forward_170(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1a = "Sub-task 1a: Identify and explicitly verify the chemical structure and functional group of each substituent attached to the benzene ring in substances 1-6, ensuring correct classification (e.g., distinguish ester group in substance 2 as C6H5–COOC2H5, chloro substituent in substance 3 as C6H5–Cl). Provide detailed chemical notation and structural descriptors."
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1a, answer1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, verifying substituent structures, thinking: {thinking1a.content}; answer: {answer1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc1a['response'] = {
        "thinking": thinking1a,
        "answer": answer1a
    }
    logs.append(subtask_desc1a)
    print("Step 1a: ", sub_tasks[-1])
    
    cot_instruction_1b = "Sub-task 1b: Determine the electronic nature (electron-donating group (EDG) or electron-withdrawing group (EWG)) of each verified substituent from subtask_1a, incorporating resonance and inductive effects, with special attention to groups like chloro and ester. Provide detailed chemical reasoning."
    N1b = self.max_sc
    cot_agents_1b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1b)]
    possible_answers_1b = []
    thinkingmapping_1b = {}
    answermapping_1b = {}
    subtask_desc1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N1b):
        thinking1b, answer1b = await cot_agents_1b[i]([taskInfo, thinking1a, answer1a], cot_instruction_1b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1b[i].id}, determining electronic nature, thinking: {thinking1b.content}; answer: {answer1b.content}")
        possible_answers_1b.append(answer1b.content)
        thinkingmapping_1b[answer1b.content] = thinking1b
        answermapping_1b[answer1b.content] = answer1b
    answer1b_content = Counter(possible_answers_1b).most_common(1)[0][0]
    thinking1b = thinkingmapping_1b[answer1b_content]
    answer1b = answermapping_1b[answer1b_content]
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc1b['response'] = {
        "thinking": thinking1b,
        "answer": answer1b
    }
    logs.append(subtask_desc1b)
    print("Step 1b: ", sub_tasks[-1])
    
    cot_instruction_1c = "Sub-task 1c: Assign the directing effects (ortho/para-directing or meta-directing) of each substituent based on their electronic nature and chemical context, explicitly noting exceptions such as halogens (chloro) which are ortho/para-directing despite being electron-withdrawing. Provide detailed justification."
    cot_agent_1c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_instruction_1c,
        "context": ["user query", "thinking of subtask 1b", "answer of subtask 1b"],
        "agent_collaboration": "CoT"
    }
    thinking1c, answer1c = await cot_agent_1c([taskInfo, thinking1b, answer1b], cot_instruction_1c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1c.id}, assigning directing effects, thinking: {thinking1c.content}; answer: {answer1c.content}")
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking1c.content}; answer - {answer1c.content}")
    subtask_desc1c['response'] = {
        "thinking": thinking1c,
        "answer": answer1c
    }
    logs.append(subtask_desc1c)
    print("Step 1c: ", sub_tasks[-1])
    
    cot_sc_instruction_1d = "Sub-task 1d: Cross-verify the substituent classifications and directing effects by generating alternative hypotheses and applying chemical consistency checks to confirm or correct assignments before proceeding."
    N1d = self.max_sc
    cot_agents_1d = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1d)]
    possible_answers_1d = []
    thinkingmapping_1d = {}
    answermapping_1d = {}
    subtask_desc1d = {
        "subtask_id": "subtask_1d",
        "instruction": cot_sc_instruction_1d,
        "context": ["user query", "thinking of subtask 1c", "answer of subtask 1c"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N1d):
        thinking1d, answer1d = await cot_agents_1d[i]([taskInfo, thinking1c, answer1c], cot_sc_instruction_1d, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1d[i].id}, cross-verifying substituent classifications, thinking: {thinking1d.content}; answer: {answer1d.content}")
        possible_answers_1d.append(answer1d.content)
        thinkingmapping_1d[answer1d.content] = thinking1d
        answermapping_1d[answer1d.content] = answer1d
    answer1d_content = Counter(possible_answers_1d).most_common(1)[0][0]
    thinking1d = thinkingmapping_1d[answer1d_content]
    answer1d = answermapping_1d[answer1d_content]
    sub_tasks.append(f"Sub-task 1d output: thinking - {thinking1d.content}; answer - {answer1d.content}")
    subtask_desc1d['response'] = {
        "thinking": thinking1d,
        "answer": answer1d
    }
    logs.append(subtask_desc1d)
    print("Step 1d: ", sub_tasks[-1])
    
    cot_instruction_2a = "Sub-task 2a: Analyze the expected para-isomer yield for each substance based on the confirmed directing effects and electronic nature, considering steric hindrance and empirical data on para/ortho/meta substitution ratios where available; categorize substituents into zero para yield and nonzero para yield groups."
    cot_agent_2a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2a = self.max_round
    cot_inputs_2a = [taskInfo, thinking1d, answer1d]
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_instruction_2a,
        "context": ["user query", "thinking of subtask 1d", "answer of subtask 1d"],
        "agent_collaboration": "Reflexion"
    }
    thinking2a, answer2a = await cot_agent_2a(cot_inputs_2a, cot_instruction_2a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2a.id}, analyzing para-isomer yield and categorizing substituents, thinking: {thinking2a.content}; answer: {answer2a.content}")
    for i in range(N_max_2a):
        feedback, correct = await critic_agent_2a([taskInfo, thinking2a, answer2a], "please review the para-isomer yield analysis and categorization and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2a.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2a.extend([thinking2a, answer2a, feedback])
        thinking2a, answer2a = await cot_agent_2a(cot_inputs_2a, cot_instruction_2a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2a.id}, refining para-isomer yield analysis, thinking: {thinking2a.content}; answer: {answer2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {
        "thinking": thinking2a,
        "answer": answer2a
    }
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])
    
    cot_instruction_2b = "Sub-task 2b: Rank the substances with nonzero para-isomer yield in order of increasing para-isomer weight fraction, incorporating quantitative or empirical data and considering relative directing strengths and steric effects."
    N2b = self.max_sc
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2b)]
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_instruction_2b,
        "context": ["user query", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2b):
        thinking2b, answer2b = await cot_agents_2b[i]([taskInfo, thinking2a, answer2a], cot_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, ranking nonzero para yield substances, thinking: {thinking2b.content}; answer: {answer2b.content}")
        possible_answers_2b.append(answer2b.content)
        thinkingmapping_2b[answer2b.content] = thinking2b
        answermapping_2b[answer2b.content] = answer2b
    answer2b_content = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking2b = thinkingmapping_2b[answer2b_content]
    answer2b = answermapping_2b[answer2b_content]
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {
        "thinking": thinking2b,
        "answer": answer2b
    }
    logs.append(subtask_desc2b)
    print("Step 2b: ", sub_tasks[-1])
    
    cot_instruction_2c = "Sub-task 2c: Rank the substances with zero para-isomer yield by their relative deactivation strength to complete the overall ordering of all substances by increasing para-isomer yield."
    cot_agent_2c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2c = {
        "subtask_id": "subtask_2c",
        "instruction": cot_instruction_2c,
        "context": ["user query", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "CoT"
    }
    thinking2c, answer2c = await cot_agent_2c([taskInfo, thinking2a, answer2a], cot_instruction_2c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2c.id}, ranking zero para yield substances by deactivation strength, thinking: {thinking2c.content}; answer: {answer2c.content}")
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking2c.content}; answer - {answer2c.content}")
    subtask_desc2c['response'] = {
        "thinking": thinking2c,
        "answer": answer2c
    }
    logs.append(subtask_desc2c)
    print("Step 2c: ", sub_tasks[-1])
    
    cot_instruction_2d = "Sub-task 2d: Integrate the rankings from subtasks 2b and 2c to produce a final ordered list of substances 1-6 by increasing para-isomer yield."
    cot_agent_2d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2d = {
        "subtask_id": "subtask_2d",
        "instruction": cot_instruction_2d,
        "context": ["user query", "thinking of subtask 2b", "answer of subtask 2b", "thinking of subtask 2c", "answer of subtask 2c"],
        "agent_collaboration": "CoT"
    }
    thinking2d, answer2d = await cot_agent_2d([taskInfo, thinking2b, answer2b, thinking2c, answer2c], cot_instruction_2d, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2d.id}, integrating rankings into final order, thinking: {thinking2d.content}; answer: {answer2d.content}")
    sub_tasks.append(f"Sub-task 2d output: thinking - {thinking2d.content}; answer - {answer2d.content}")
    subtask_desc2d['response'] = {
        "thinking": thinking2d,
        "answer": answer2d
    }
    logs.append(subtask_desc2d)
    print("Step 2d: ", sub_tasks[-1])
    
    debate_instruction_3 = "Sub-task 3: Compare the final ordered list from subtask_2d with the provided multiple-choice options and select the correct choice that matches the order of increasing para-isomer yield."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", "thinking of subtask 2d", "answer of subtask 2d"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2d, answer2d], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2d, answer2d] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting correct multiple-choice option, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Make final decision on the correct multiple-choice option.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final multiple-choice option, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs