async def forward_167(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_1 = (
        "Sub-task 1: Analyze and characterize each of the four issues (mutually incompatible data formats, 'chr' / 'no chr' confusion, "
        "reference assembly mismatch, incorrect ID conversion) by explicitly identifying their typical failure modes in genomics data analysis pipelines. "
        "Include concrete examples where each issue causes immediate, obvious errors versus subtle, difficult-to-spot erroneous results. "
        "Incorporate domain-specific evidence to avoid the flawed assumption that incompatible data formats always cause explicit failures, "
        "and instead explore scenarios where format issues are silently tolerated or partially auto-corrected, leading to subtle errors. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )

    debate_agents_1 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_1 = self.max_round

    all_thinking_1 = [[] for _ in range(N_max_1)]
    all_answer_1 = [[] for _ in range(N_max_1)]

    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking1, answer1 = await agent([taskInfo], debate_instr_1, r, is_sub_task=True)
            else:
                input_infos_1 = [taskInfo] + all_thinking_1[r-1] + all_answer_1[r-1]
                thinking1, answer1 = await agent(input_infos_1, debate_instr_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing issues, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking_1[r].append(thinking1)
            all_answer_1[r].append(answer1)

    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1(
        [taskInfo] + all_thinking_1[-1] + all_answer_1[-1],
        "Sub-task 1: Synthesize and finalize analysis and characterization of the four issues." +
        "Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )

    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    reflect_inst = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction = (
        "Sub-task 2: Conduct a Reflexion step to critically revisit and challenge all major assumptions made in Subtask 1, "
        "especially regarding the subtlety and detectability of errors caused by mutually incompatible data formats. "
        "Use domain-specific evidence and real-world genomics examples to validate or refute these assumptions. "
        "The goal is to reconcile conflicting views and ensure a shared, accurate understanding of which issues cause difficult-to-spot errors versus immediate failures before proceeding. "
        + reflect_inst
    )

    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2 = self.max_round

    cot_inputs_2 = [taskInfo, thinking1, answer1]

    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_reflect_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Reflexion"
    }

    thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, revisiting assumptions, thinking: {thinking2.content}; answer: {answer2.content}")

    critic_inst_2 = (
        "Please review the answer above and criticize on where might be wrong. "
        "If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    )

    for i in range(N_max_2):
        feedback, correct = await critic_agent_2(
            [taskInfo, thinking2, answer2],
            "Please review and provide the limitations of provided solutions." + critic_inst_2,
            i,
            is_sub_task=True
        )
        agents.append(f"Critic agent {critic_agent_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining assumptions, thinking: {thinking2.content}; answer: {answer2.content}")

    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_3 = (
        "Sub-task 3: Assess the frequency and commonality of each issue in real-world genomics data analysis workflows, "
        "focusing on which issues are most prevalent and likely to cause subtle, difficult-to-spot erroneous results. "
        "Leverage the refined understanding from Subtasks 1 and 2, ensuring frequency estimates are consistent with the clarified failure mode distinctions. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )

    debate_agents_3 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_3 = self.max_round

    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]

    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instr_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking1, answer1, thinking2, answer2], debate_instr_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instr_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, assessing frequency, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_3[r].append(thinking3)
            all_answer_3[r].append(answer3)

    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3(
        [taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking_3[-1] + all_answer_3[-1],
        "Sub-task 3: Synthesize and finalize frequency and commonality assessment." +
        "Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )

    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    debate_instr_4 = (
        "Sub-task 4: Evaluate the combined impact of the issues when occurring together, emphasizing which combinations are known to be the most common sources of difficult-to-spot errors in genomics data analysis. "
        "Integrate insights from the characterization, reflexion, and frequency assessment subtasks to avoid contradictory conclusions and ensure a coherent synthesis. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )

    debate_agents_4 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_4 = self.max_round

    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]

    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instr_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instr_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instr_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating combined impact, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking_4[r].append(thinking4)
            all_answer_4[r].append(answer4)

    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4(
        [taskInfo, thinking3, answer3] + all_thinking_4[-1] + all_answer_4[-1],
        "Sub-task 4: Synthesize and finalize evaluation of combined impact." +
        "Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )

    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instr_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate",
        "response": {"thinking": thinking4, "answer": answer4}
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_5 = (
        "Sub-task 5: Select and justify the best answer choice from the given options (3 and 4, 2 and 3, all of the above, 2, 3 and 4) based on the comprehensive analysis and evaluation of issue subtlety, frequency, and combined impact. "
        "The justification should explicitly reference the resolved assumptions and evidence gathered in prior subtasks to support the final decision."
    )

    N_sc = self.max_sc
    cot_agents_5 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc)
    ]

    possible_answers_5 = []
    possible_thinkings_5 = []

    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "SC_CoT"
    }

    for i in range(N_sc):
        thinking5, answer5 = await cot_agents_5[i](
            [taskInfo, thinking4, answer4],
            cot_sc_instruction_5,
            is_sub_task=True
        )
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, selecting best answer, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5)
        possible_thinkings_5.append(thinking5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5(
        [taskInfo, thinking4, answer4] + possible_thinkings_5 + possible_answers_5,
        "Sub-task 5: Synthesize and choose the most consistent and correct answer choice." +
        "Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )

    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
