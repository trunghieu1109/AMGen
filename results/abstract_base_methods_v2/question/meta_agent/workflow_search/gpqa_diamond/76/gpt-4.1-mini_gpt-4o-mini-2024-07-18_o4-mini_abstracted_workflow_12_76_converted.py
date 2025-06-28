async def forward_76(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_temperature = 0.0
    sc_temperature = 0.5
    max_sc = self.max_sc
    max_round = self.max_round

    # Stage 1: Compute and Apply Transformations

    # Subtask 1a: Classify the reaction type for the first reaction
    instruction_1a = ("Subtask 1a: Classify the reaction type for the first reaction (((3-methylbut-2-en-1-yl)oxy)methyl)benzene treated with BuLi followed by H+. "
                      "Analyze the substrate structure, reagents, and conditions to distinguish between possible rearrangements such as Claisen versus Cope rearrangement. "
                      "Provide detailed mechanistic guidelines and reasoning.")
    agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature)
    desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1a, answer_1a = await agent_1a([taskInfo], instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {agent_1a.id}, classifying reaction type, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Subtask 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    desc_1a['response'] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(desc_1a)
    print("Step 1a: ", sub_tasks[-1])

    # Subtask 1b: Generate detailed mechanistic pathways for each plausible rearrangement
    instruction_1b = ("Subtask 1b: Generate detailed step-by-step mechanistic pathways for each plausible rearrangement identified in Subtask 1a, "
                      "including electron flow, intermediate structures, and stereochemical considerations.")
    agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature)
    desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": instruction_1b,
        "context": ["user query", thinking_1a, answer_1a],
        "agent_collaboration": "CoT"
    }
    thinking_1b, answer_1b = await agent_1b([taskInfo, thinking_1a, answer_1a], instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {agent_1b.id}, generating mechanistic pathways, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
    sub_tasks.append(f"Subtask 1b output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    desc_1b['response'] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(desc_1b)
    print("Step 1b: ", sub_tasks[-1])

    # Subtask 1c: Propose all plausible major products from the first reaction
    instruction_1c = ("Subtask 1c: Propose all plausible major products from the first reaction based on the mechanistic pathways from Subtask 1b, "
                      "including isomeric and stereochemical variants.")
    agent_1c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature)
    desc_1c = {
        "subtask_id": "subtask_1c",
        "instruction": instruction_1c,
        "context": ["user query", thinking_1b, answer_1b],
        "agent_collaboration": "CoT"
    }
    thinking_1c, answer_1c = await agent_1c([taskInfo, thinking_1b, answer_1b], instruction_1c, is_sub_task=True)
    agents.append(f"CoT agent {agent_1c.id}, proposing plausible products, thinking: {thinking_1c.content}; answer: {answer_1c.content}")
    sub_tasks.append(f"Subtask 1c output: thinking - {thinking_1c.content}; answer - {answer_1c.content}")
    desc_1c['response'] = {"thinking": thinking_1c, "answer": answer_1c}
    logs.append(desc_1c)
    print("Step 1c: ", sub_tasks[-1])

    # Subtask 1d: Evaluate and rank the proposed products
    instruction_1d = ("Subtask 1d: Evaluate and rank the proposed products from Subtask 1c based on mechanistic plausibility, thermodynamic stability, "
                      "and stereochemical factors to select the most likely major product A.")
    agent_1d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature)
    desc_1d = {
        "subtask_id": "subtask_1d",
        "instruction": instruction_1d,
        "context": ["user query", thinking_1c, answer_1c],
        "agent_collaboration": "CoT"
    }
    thinking_1d, answer_1d = await agent_1d([taskInfo, thinking_1c, answer_1c], instruction_1d, is_sub_task=True)
    agents.append(f"CoT agent {agent_1d.id}, evaluating and ranking products, thinking: {thinking_1d.content}; answer: {answer_1d.content}")
    sub_tasks.append(f"Subtask 1d output: thinking - {thinking_1d.content}; answer - {answer_1d.content}")
    desc_1d['response'] = {"thinking": thinking_1d, "answer": answer_1d}
    logs.append(desc_1d)
    print("Step 1d: ", sub_tasks[-1])

    # Subtask 1e: Verification and debate for first reaction
    instruction_1e = ("Subtask 1e: Conduct a verification and debate stage for the first reaction by comparing alternative mechanistic pathways and product proposals, "
                      "using multiple reasoning paths or agents to reach consensus on the most chemically sound product A.")
    debate_agents_1e = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=sc_temperature) for role in self.debate_role]
    N_max_1e = max_round
    all_thinking_1e = [[] for _ in range(N_max_1e)]
    all_answer_1e = [[] for _ in range(N_max_1e)]
    desc_1e = {
        "subtask_id": "subtask_1e",
        "instruction": instruction_1e,
        "context": ["user query", thinking_1d, answer_1d],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1e):
        for i, agent in enumerate(debate_agents_1e):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_1d, answer_1d], instruction_1e, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1d, answer_1d] + all_thinking_1e[r-1] + all_answer_1e[r-1]
                thinking, answer = await agent(input_infos, instruction_1e, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating first reaction products, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_1e[r].append(thinking)
            all_answer_1e[r].append(answer)
    final_decision_agent_1e = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=cot_temperature)
    thinking_1e, answer_1e = await final_decision_agent_1e([taskInfo] + all_thinking_1e[-1] + all_answer_1e[-1], "Subtask 1e: Make final decision on the most plausible major product A.", is_sub_task=True)
    agents.append(f"Final Decision agent for first reaction, thinking: {thinking_1e.content}; answer: {answer_1e.content}")
    sub_tasks.append(f"Subtask 1e output: thinking - {thinking_1e.content}; answer - {answer_1e.content}")
    desc_1e['response'] = {"thinking": thinking_1e, "answer": answer_1e}
    logs.append(desc_1e)
    print("Step 1e: ", sub_tasks[-1])

    # Stage 2: Evaluate Conformity and Validity

    # Subtask 2a: Analyze the second reaction to confirm Cope rearrangement
    instruction_2a = ("Subtask 2a: Analyze the second reaction (3,4,5,7,8,9-hexamethyl-1,11-dimethylene-2,6,10,11,11a,11b-hexahydro-1H-benzo[cd]indeno[7,1-gh]azulene + Heat) "
                      "to confirm the Cope rearrangement mechanism, detailing the rearrangement steps, intermediate structures, and stereochemical outcomes.")
    agent_2a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature)
    desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": instruction_2a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_2a, answer_2a = await agent_2a([taskInfo], instruction_2a, is_sub_task=True)
    agents.append(f"CoT agent {agent_2a.id}, analyzing second reaction mechanism, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
    sub_tasks.append(f"Subtask 2a output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    desc_2a['response'] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(desc_2a)
    print("Step 2a: ", sub_tasks[-1])

    # Subtask 2b: Propose major product B from second reaction
    instruction_2b = ("Subtask 2b: Propose the major product B from the second reaction based on the Cope rearrangement mechanism analyzed in Subtask 2a, "
                      "including possible isomers and stereochemical variants.")
    agent_2b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature)
    desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": instruction_2b,
        "context": ["user query", thinking_2a, answer_2a],
        "agent_collaboration": "CoT"
    }
    thinking_2b, answer_2b = await agent_2b([taskInfo, thinking_2a, answer_2a], instruction_2b, is_sub_task=True)
    agents.append(f"CoT agent {agent_2b.id}, proposing product B, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
    sub_tasks.append(f"Subtask 2b output: thinking - {thinking_2b.content}; answer - {answer_2b.content}")
    desc_2b['response'] = {"thinking": thinking_2b, "answer": answer_2b}
    logs.append(desc_2b)
    print("Step 2b: ", sub_tasks[-1])

    # Subtask 2c: Evaluate and rank proposed products from Subtask 2b
    instruction_2c = ("Subtask 2c: Evaluate and rank the proposed products from Subtask 2b based on mechanistic plausibility, thermodynamic stability, "
                      "and stereochemical considerations to select the most likely major product B.")
    agent_2c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature)
    desc_2c = {
        "subtask_id": "subtask_2c",
        "instruction": instruction_2c,
        "context": ["user query", thinking_2b, answer_2b],
        "agent_collaboration": "CoT"
    }
    thinking_2c, answer_2c = await agent_2c([taskInfo, thinking_2b, answer_2b], instruction_2c, is_sub_task=True)
    agents.append(f"CoT agent {agent_2c.id}, evaluating and ranking products B, thinking: {thinking_2c.content}; answer: {answer_2c.content}")
    sub_tasks.append(f"Subtask 2c output: thinking - {thinking_2c.content}; answer - {answer_2c.content}")
    desc_2c['response'] = {"thinking": thinking_2c, "answer": answer_2c}
    logs.append(desc_2c)
    print("Step 2c: ", sub_tasks[-1])

    # Subtask 2d: Verification and debate for second reaction
    instruction_2d = ("Subtask 2d: Perform a verification and debate stage for the second reaction by generating multiple reasoning paths and cross-validating the proposed product B "
                      "to ensure chemical soundness and consistency.")
    debate_agents_2d = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=sc_temperature) for role in self.debate_role]
    N_max_2d = max_round
    all_thinking_2d = [[] for _ in range(N_max_2d)]
    all_answer_2d = [[] for _ in range(N_max_2d)]
    desc_2d = {
        "subtask_id": "subtask_2d",
        "instruction": instruction_2d,
        "context": ["user query", thinking_2c, answer_2c],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2d):
        for i, agent in enumerate(debate_agents_2d):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_2c, answer_2c], instruction_2d, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2c, answer_2c] + all_thinking_2d[r-1] + all_answer_2d[r-1]
                thinking, answer = await agent(input_infos, instruction_2d, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating second reaction products, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_2d[r].append(thinking)
            all_answer_2d[r].append(answer)
    final_decision_agent_2d = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=cot_temperature)
    thinking_2d, answer_2d = await final_decision_agent_2d([taskInfo] + all_thinking_2d[-1] + all_answer_2d[-1], "Subtask 2d: Make final decision on the most plausible major product B.", is_sub_task=True)
    agents.append(f"Final Decision agent for second reaction, thinking: {thinking_2d.content}; answer: {answer_2d.content}")
    sub_tasks.append(f"Subtask 2d output: thinking - {thinking_2d.content}; answer - {answer_2d.content}")
    desc_2d['response'] = {"thinking": thinking_2d, "answer": answer_2d}
    logs.append(desc_2d)
    print("Step 2d: ", sub_tasks[-1])

    # Stage 3: Evaluate and Prioritize Elements

    # Subtask 3: Cross-validate products A and B for consistency
    instruction_3 = ("Subtask 3: Cross-validate the selected major products A and B from Subtasks 1e and 2d to check for internal consistency and chemical plausibility. "
                     "If contradictions or uncertainties arise, revisit and refine earlier subtasks.")
    agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature)
    critic_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=cot_temperature)
    cot_inputs_3 = [taskInfo, thinking_1e, answer_1e, thinking_2d, answer_2d]
    desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": instruction_3,
        "context": ["user query", thinking_1e, answer_1e, thinking_2d, answer_2d],
        "agent_collaboration": "Reflexion"
    }
    thinking_3, answer_3 = await agent_3(cot_inputs_3, instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {agent_3.id}, cross-validating products A and B, thinking: {thinking_3.content}; answer: {answer_3.content}")
    for i in range(max_round):
        feedback, correct = await critic_3([taskInfo, thinking_3, answer_3], "Please review the cross-validation and identify any inconsistencies or errors.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking_3, answer_3, feedback])
        thinking_3, answer_3 = await agent_3(cot_inputs_3, instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {agent_3.id}, refining cross-validation, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(desc_3)
    print("Step 3: ", sub_tasks[-1])

    # Subtask 4: Compare final verified products with multiple-choice options
    instruction_4 = ("Subtask 4: Compare the final verified products A and B with the given multiple-choice options to identify the correct choice that matches both products accurately.")
    agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature)
    critic_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=cot_temperature)
    cot_inputs_4 = [taskInfo, thinking_1e, answer_1e, thinking_2d, answer_2d, thinking_3, answer_3]
    desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": instruction_4,
        "context": ["user query", thinking_1e, answer_1e, thinking_2d, answer_2d, thinking_3, answer_3],
        "agent_collaboration": "Reflexion"
    }
    thinking_4, answer_4 = await agent_4(cot_inputs_4, instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {agent_4.id}, comparing final products with choices, thinking: {thinking_4.content}; answer: {answer_4.content}")
    for i in range(max_round):
        feedback, correct = await critic_4([taskInfo, thinking_4, answer_4], "Please review the comparison with choices and identify any inconsistencies or errors.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking_4, answer_4, feedback])
        thinking_4, answer_4 = await agent_4(cot_inputs_4, instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {agent_4.id}, refining comparison with choices, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(desc_4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_4, answer_4, sub_tasks, agents)
    return final_answer, logs
