async def forward_76(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_sc_agents_num = self.max_sc
    debate_rounds = self.max_round
    debate_roles = self.debate_role
    
    cot_instruction_1a = (
        "Sub-task 1a: Enumerate and evaluate all plausible reaction mechanisms for the transformation of (((3-methylbut-2-en-1-yl)oxy)methyl)benzene with (1. BuLi, 2. H+), "
        "including but not limited to deprotonation, [2,3]-Wittig rearrangement, and SN2' pathways. For each mechanism, provide detailed mechanistic steps, intermediate structures, and expected stereochemical outcomes. "
        "Include structural depictions or SMILES if possible, and cite relevant literature or named reactions. Rate confidence levels for each mechanism."
    )
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    cot_agents_1a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_agents_num)]
    possible_answers_1a = []
    thinkingmapping_1a = {}
    answermapping_1a = {}
    for i in range(cot_sc_agents_num):
        thinking1a, answer1a = await cot_agents_1a[i]([taskInfo], cot_instruction_1a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1a[i].id}, enumerating mechanisms, thinking: {thinking1a.content}; answer: {answer1a.content}")
        possible_answers_1a.append(answer1a.content)
        thinkingmapping_1a[answer1a.content] = thinking1a
        answermapping_1a[answer1a.content] = answer1a
    most_common_answer_1a = Counter(possible_answers_1a).most_common(1)[0][0]
    thinking1a = thinkingmapping_1a[most_common_answer_1a]
    answer1a = answermapping_1a[most_common_answer_1a]
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking1a, "answer": answer1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])
    
    cot_instruction_1b = (
        "Sub-task 1b: Based on the mechanistic hypotheses generated in Subtask 1a, predict the major product(s) A, including regiochemistry and stereochemistry. "
        "Assess the relative likelihood of each product using chemical reasoning, stereochemical detail, and literature precedents. Provide detailed justifications and confidence ratings."
    )
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a"],
        "agent_collaboration": "SC_CoT"
    }
    cot_agents_1b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_agents_num)]
    possible_answers_1b = []
    thinkingmapping_1b = {}
    answermapping_1b = {}
    for i in range(cot_sc_agents_num):
        thinking1b, answer1b = await cot_agents_1b[i]([taskInfo, thinking1a, answer1a], cot_instruction_1b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1b[i].id}, predicting product A, thinking: {thinking1b.content}; answer: {answer1b.content}")
        possible_answers_1b.append(answer1b.content)
        thinkingmapping_1b[answer1b.content] = thinking1b
        answermapping_1b[answer1b.content] = answer1b
    most_common_answer_1b = Counter(possible_answers_1b).most_common(1)[0][0]
    thinking1b = thinkingmapping_1b[most_common_answer_1b]
    answer1b = answermapping_1b[most_common_answer_1b]
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])
    
    cot_instruction_2a = (
        "Sub-task 2a: Analyze the starting material 3,4,5,7,8,9-hexamethyl-1,11-dimethylene-2,6,10,11,11a,11b-hexahydro-1H-benzo[cd]indeno[7,1-gh]azulene and the effect of heat to enumerate all plausible Cope rearrangement pathways, "
        "including alternative ring conformations and stereochemical outcomes. Provide detailed mechanistic rationales, intermediate structures, and stereochemical assignments. Include literature references and confidence ratings."
    )
    subtask_desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_instruction_2a,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_agents_num)]
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    for i in range(cot_sc_agents_num):
        thinking2a, answer2a = await cot_agents_2a[i]([taskInfo], cot_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, enumerating Cope pathways, thinking: {thinking2a.content}; answer: {answer2a.content}")
        possible_answers_2a.append(answer2a.content)
        thinkingmapping_2a[answer2a.content] = thinking2a
        answermapping_2a[answer2a.content] = answer2a
    most_common_answer_2a = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking2a = thinkingmapping_2a[most_common_answer_2a]
    answer2a = answermapping_2a[most_common_answer_2a]
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking2a, "answer": answer2a}
    logs.append(subtask_desc_2a)
    print("Step 2a: ", sub_tasks[-1])
    
    cot_instruction_2b = (
        "Sub-task 2b: Predict the major product(s) B resulting from the Cope rearrangement(s) analyzed in Subtask 2a, specifying regiochemistry, stereochemistry, and ring system modifications. "
        "Evaluate alternative products and rank their likelihood based on mechanistic and stereochemical considerations with detailed justifications and confidence levels."
    )
    subtask_desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_instruction_2b,
        "context": ["user query", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "SC_CoT"
    }
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_agents_num)]
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    for i in range(cot_sc_agents_num):
        thinking2b, answer2b = await cot_agents_2b[i]([taskInfo, thinking2a, answer2a], cot_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, predicting product B, thinking: {thinking2b.content}; answer: {answer2b.content}")
        possible_answers_2b.append(answer2b.content)
        thinkingmapping_2b[answer2b.content] = thinking2b
        answermapping_2b[answer2b.content] = answer2b
    most_common_answer_2b = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking2b = thinkingmapping_2b[most_common_answer_2b]
    answer2b = answermapping_2b[most_common_answer_2b]
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc_2b['response'] = {"thinking": thinking2b, "answer": answer2b}
    logs.append(subtask_desc_2b)
    print("Step 2b: ", sub_tasks[-1])
    
    reflexion_instruction_3 = (
        "Sub-task 3: Reflexion stage: Critically review the outputs from Subtasks 1b and 2b to identify inconsistencies, overlooked alternative mechanisms, or stereochemical outcomes. "
        "If inconsistencies or doubts arise, request re-analysis or refinement of earlier subtasks. Provide detailed reasoning and confidence assessments."
    )
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": reflexion_instruction_3,
        "context": ["user query", "thinking of subtask 1b", "answer of subtask 1b", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_3 = [taskInfo, thinking1b, answer1b, thinking2b, answer2b]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, reflexion_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, reviewing products A and B, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        feedback3, correct3 = await critic_agent_3([taskInfo, thinking3, answer3], "Please review the reflexion output and provide limitations or corrections.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, reflexion_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining reflexion, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])
    
    debate_instruction_4 = (
        "Sub-task 4: Compare the final consensus predicted products A and B with each of the provided multiple-choice options. "
        "Evaluate the match in terms of structure, regiochemistry, and stereochemistry, debate the merits of each choice, and select the choice that best corresponds to both products. "
        "Incorporate reflections and alternative hypotheses from previous stages to robustly finalize the answer."
    )
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    all_thinking4 = [[] for _ in range(debate_rounds)]
    all_answer4 = [[] for _ in range(debate_rounds)]
    for r in range(debate_rounds):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating choices, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the correct choice matching products A and B.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct choice, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])
    
    crosscheck_instruction_5 = (
        "Sub-task 5: Perform a cross-check validation of the selected answer from Subtask 4 against established chemical reaction databases, literature precedents, and known reaction rules. "
        "Confirm the correctness and robustness of the final selection, and identify any mismatches or inconsistencies. Provide detailed reasoning and confidence levels."
    )
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": crosscheck_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "CoT"
    }
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], crosscheck_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, cross-checking final answer, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc_5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
