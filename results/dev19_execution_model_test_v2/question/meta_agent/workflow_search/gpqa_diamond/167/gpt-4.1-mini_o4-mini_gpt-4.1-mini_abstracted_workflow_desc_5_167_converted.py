async def forward_167(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Extract and Analyze Issues

    # Subtask 1: Extract and Summarize Defining Features with SC_CoT
    cot_sc_instruction = (
        "Sub-task 1: Extract and comprehensively summarize the defining features, characteristics, "
        "and real-world evidence of subtle and silent failures for each of the four issues in genomics data analysis: "
        "mutually incompatible data formats, 'chr' / 'no chr' confusion, reference assembly mismatch, and incorrect ID conversion. "
        "Explicitly address documented examples where format incompatibilities slip through validation and cause difficult-to-spot downstream errors."
    )
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    possible_thinkings = []
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1, answer1 = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, extracting and summarizing issues, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers.append(answer1)
        possible_thinkings.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1(
        [taskInfo] + possible_thinkings + possible_answers,
        "Sub-task 1: Synthesize and choose the most consistent and comprehensive summary of the four issues with real-world evidence.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Subtask 2: Analyze and Classify Issues with Debate
    debate_instr_2 = (
        "Sub-task 2: Analyze and classify all four issues based on their frequency, subtlety (difficulty to detect), "
        "and impact on genomics data analysis accuracy, using evidence and examples from Subtask 1. "
        "Avoid flawed assumptions by critically considering nuanced and partial format incompatibilities. "
        "Integrate cross-issue comparisons and prepare a balanced assessment for subsequent reflexion. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking2 = [[] for _ in range(N_max_2)]
    all_answer2 = [[] for _ in range(N_max_2)]
    subtask_desc2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instr_2,
        "context": ["user query", thinking1, answer1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instr_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
                thinking2, answer2 = await agent(input_infos_2, debate_instr_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing and classifying issues, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2(
        [taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1],
        "Sub-task 2: Synthesize and choose the most consistent classification of issues based on frequency, subtlety, and impact.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Subtask 3: Reflexion and Adversarial Review
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = (
        "Sub-task 3: Conduct a Reflexion and adversarial review of the assumptions and conclusions from Subtasks 1 and 2, "
        "focusing on challenging any premature exclusion or underestimation of issues—especially 'mutually incompatible data formats'. "
        "Explicitly re-examine detectability claims and frequency assessments to ensure no issue is overlooked or mischaracterized. "
        + reflect_inst
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_reflect_instruction,
        "context": ["user query", thinking1, answer1.content, thinking2, answer2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, reviewing assumptions and conclusions, thinking: {thinking3.content}; answer: {answer3.content}")
    critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining conclusions, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Map Classification to Answer Choices

    debate_instr_4 = (
        "Sub-task 1 (Stage 2): Transform the refined and validated classification of issues from Stage 1 into variant groupings corresponding to the provided answer choices. "
        "Carefully map the evidence-based assessment of frequency and subtlety to the combinations of issues in the answer options, ensuring all common and difficult-to-spot issues are included. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instr_4,
        "context": ["user query", thinking3, answer3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instr_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instr_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, mapping classification to answer choices, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4(
        [taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1],
        "Sub-task 1 (Stage 2): Synthesize and choose the best mapping of issues to provided answer choices.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 3: Evaluate and Prioritize Answer Choices

    debate_instr_5 = (
        "Sub-task 1 (Stage 3): Evaluate and prioritize the variant groupings generated in Stage 2 against the criteria of commonality and subtlety of errors, "
        "using the comprehensive evidence and reflexion outputs. Select the best answer choice that accurately reflects the most common sources of difficult-to-spot erroneous results in genomics data analysis, "
        "explicitly avoiding prior mistakes of excluding subtle format incompatibilities. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instr_5,
        "context": ["user query", thinking4, answer4.content, thinking3, answer3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4, thinking3, answer3], debate_instr_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4, thinking3, answer3] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instr_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating and prioritizing answer choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5(
        [taskInfo, thinking4, answer4, thinking3, answer3] + all_thinking5[-1] + all_answer5[-1],
        "Sub-task 1 (Stage 3): Select the best answer choice reflecting the most common and difficult-to-spot erroneous sources in genomics data analysis.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, calculating final answer, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Stage 3 Sub-task 1 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
