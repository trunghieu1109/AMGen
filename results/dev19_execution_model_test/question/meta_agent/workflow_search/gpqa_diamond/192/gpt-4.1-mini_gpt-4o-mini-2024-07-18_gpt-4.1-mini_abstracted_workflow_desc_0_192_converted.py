async def forward_192(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0 = (
        "Sub-task 1: Analyze and classify the given information: the star count dependence on parallax (1/plx^5), "
        "the inverse relation between parallax and distance (plx ~ 1/r), and the meaning of star counts per unit parallax interval, "
        "with context from the user query."
    )
    cot_agents_0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_0 = []
    possible_thinkings_0 = []
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking0, answer0 = await cot_agents_0[i]([taskInfo], cot_sc_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0[i].id}, analyzing given info, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_thinkings_0.append(thinking0)
        possible_answers_0.append(answer0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + possible_thinkings_0 + possible_answers_0, "Sub-task 1: Synthesize and choose the most consistent analysis of given information.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc_0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_1 = (
        "Sub-task 2: Assess the impact of substituting parallax with distance in the star count relation, "
        "including how the differential intervals (dplx and dr) relate and affect the distribution, "
        "based on the analysis from Sub-task 1 and the user query."
    )
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo, thinking0, answer0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, assessing substitution impact, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thinkings_1.append(thinking1)
        possible_answers_1.append(answer1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo, thinking0, answer0] + possible_thinkings_1 + possible_answers_1, "Sub-task 2: Synthesize and choose the most consistent assessment of substitution impact.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 3: Derive the expression for the number of stars per unit distance interval by applying the change of variables from parallax to distance, "
        "incorporating the Jacobian of the transformation, based on the assessment from Sub-task 2 and the user query."
    )
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, deriving expression with Jacobian, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_thinkings_2.append(thinking2)
        possible_answers_2.append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings_2 + possible_answers_2, "Sub-task 3: Synthesize and choose the most consistent derived expression.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3_1 = (
        "Sub-task 4.1: Combine and transform the quantitative inputs to simplify the derived expression and identify the power-law dependence of star counts on distance r, "
        "based on the derivation from Sub-task 3 and the user query."
    )
    cot_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", thinking2, answer2],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking3_1, answer3_1 = await cot_agents_3_1[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3_1[i].id}, simplifying expression, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
        possible_thinkings_3_1.append(thinking3_1)
        possible_answers_3_1.append(answer3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await final_decision_agent_3_1([taskInfo, thinking2, answer2] + possible_thinkings_3_1 + possible_answers_3_1, "Sub-task 4.1: Synthesize and choose the most consistent simplified expression.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4.1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    debate_instruction_3_2 = (
        "Sub-task 4.2: Compare the derived dependence with the given multiple-choice options and select the correct scaling law for the number of stars per unit distance interval. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_2 = self.max_round
    all_thinking_3_2 = [[] for _ in range(N_max_3_2)]
    all_answer_3_2 = [[] for _ in range(N_max_3_2)]
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": debate_instruction_3_2,
        "context": ["user query", thinking3_1, answer3_1],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_2):
        for i, agent in enumerate(debate_agents_3_2):
            if r == 0:
                thinking3_2, answer3_2 = await agent([taskInfo, thinking3_1, answer3_1], debate_instruction_3_2, r, is_sub_task=True)
            else:
                input_infos_3_2 = [taskInfo, thinking3_1, answer3_1] + all_thinking_3_2[r-1] + all_answer_3_2[r-1]
                thinking3_2, answer3_2 = await agent(input_infos_3_2, debate_instruction_3_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing with options, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
            all_thinking_3_2[r].append(thinking3_2)
            all_answer_3_2[r].append(answer3_2)
    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_2, answer3_2 = await final_decision_agent_3_2(
        [taskInfo, thinking3_1, answer3_1] + all_thinking_3_2[-1] + all_answer_3_2[-1],
        "Sub-task 4.2: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 4.2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking3_2, "answer": answer3_2}
    logs.append(subtask_desc_3_2)
    print("Step 3.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3_2, answer3_2, sub_tasks, agents)
    return final_answer, logs
