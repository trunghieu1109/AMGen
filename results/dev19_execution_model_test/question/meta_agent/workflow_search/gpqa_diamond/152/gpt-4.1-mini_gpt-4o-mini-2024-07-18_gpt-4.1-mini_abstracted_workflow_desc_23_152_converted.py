async def forward_152(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Analyze and classify all reactants, reagents, and compound C (Debate)
    debate_instr_stage1 = (
        "Sub-task 1: Analyze and classify all given reactants, reagents, and reaction conditions, "
        "explicitly including the identification and structural definition of compound C. "
        "Clarify the chemical nature and roles of each component in the Michael addition context to avoid ambiguity. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_stage1 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_stage1 = self.max_round
    all_thinking_stage1 = [[] for _ in range(N_max_stage1)]
    all_answer_stage1 = [[] for _ in range(N_max_stage1)]
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instr_stage1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage1):
        for i, agent in enumerate(debate_agents_stage1):
            if r == 0:
                thinking, answer = await agent([taskInfo], debate_instr_stage1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo] + all_thinking_stage1[r-1] + all_answer_stage1[r-1]
                thinking, answer = await agent(input_infos, debate_instr_stage1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing reactants and compound C, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_stage1[r].append(thinking)
            all_answer_stage1[r].append(answer)
    final_decision_agent_stage1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_stage1(
        [taskInfo] + all_thinking_stage1[-1] + all_answer_stage1[-1],
        "Sub-task 1: Synthesize and choose the most consistent classification and identification of reactants and compound C.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2: Apply Michael addition mechanism to derive major products (SC_CoT)
    cot_sc_instruction_stage2 = (
        "Sub-task 2: Based on the output from Stage 1 Sub-task 1, apply the Michael addition mechanism to each reaction (A, B, and C) "
        "to derive the expected major products. Include detailed mechanistic reasoning about nucleophilic attack sites, resonance stabilization, "
        "and influence of reaction conditions. Justify formation of functional groups and regiochemistry."
    )
    N_sc = self.max_sc
    cot_agents_stage2 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc)
    ]
    possible_answers_stage2 = []
    possible_thinkings_stage2 = []
    subtask_desc2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_stage2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_stage2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_stage2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage2[i].id}, deriving major products, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_stage2.append(answer2)
        possible_thinkings_stage2.append(thinking2)
    final_decision_agent_stage2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_stage2(
        [taskInfo, thinking1, answer1] + possible_thinkings_stage2 + possible_answers_stage2,
        "Sub-task 2: Synthesize and choose the most consistent major products for reactions A, B, and C.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 2 Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 3: Structural elucidation and mapping of derived products (Debate)
    debate_instr_stage3 = (
        "Sub-task 3: Perform detailed structural elucidation and mapping of the derived products from Stage 2 Sub-task 2. "
        "Number all carbons in the product skeletons, assign substituent positions precisely, and confirm functional group identities. "
        "Use reflexive checks to verify consistency between mechanistic reasoning and structural assignments. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_stage3 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_stage3 = self.max_round
    all_thinking_stage3 = [[] for _ in range(N_max_stage3)]
    all_answer_stage3 = [[] for _ in range(N_max_stage3)]
    subtask_desc3 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instr_stage3,
        "context": ["user query", thinking2, answer2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage3):
        for i, agent in enumerate(debate_agents_stage3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instr_stage3, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking2, answer2] + all_thinking_stage3[r-1] + all_answer_stage3[r-1]
                thinking3, answer3 = await agent(input_infos, debate_instr_stage3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, structural elucidation, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_stage3[r].append(thinking3)
            all_answer_stage3[r].append(answer3)
    final_decision_agent_stage3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_stage3(
        [taskInfo, thinking2, answer2] + all_thinking_stage3[-1] + all_answer_stage3[-1],
        "Sub-task 3: Synthesize and confirm structural elucidation and mapping of products.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 3 Sub-task 1 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 4: Multi-criteria evaluation of multiple-choice options (SC_CoT)
    cot_sc_instruction_stage4 = (
        "Sub-task 4: Integrate all mechanistic, structural, and contextual information from previous subtasks to perform a multi-criteria evaluation of the multiple-choice options. "
        "Reconcile any conflicting interpretations through iterative reasoning and consensus-building, ensuring the final selection is mechanistically justified and consistent with clarified structures and reaction conditions."
    )
    cot_agents_stage4 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc)
    ]
    possible_answers_stage4 = []
    possible_thinkings_stage4 = []
    subtask_desc4 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_sc_instruction_stage4,
        "context": ["user query", thinking1, answer1, thinking2, answer2, thinking3, answer3],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_stage4[i](
            [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3],
            cot_sc_instruction_stage4, is_sub_task=True
        )
        agents.append(f"CoT-SC agent {cot_agents_stage4[i].id}, evaluating multiple-choice options, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_stage4.append(answer4)
        possible_thinkings_stage4.append(thinking4)
    final_decision_agent_stage4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_stage4(
        [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3] + possible_thinkings_stage4 + possible_answers_stage4,
        "Sub-task 4: Synthesize and select the most consistent and correct multiple-choice answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 4 Sub-task 1 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
