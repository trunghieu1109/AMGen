async def forward_6(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = (
        "Sub-task 1: Precisely formulate the mathematical problem by defining variables x, y, z > 0, "
        "constraints for surface area and volume, and clarify that r is half the space diagonal length. "
        "Avoid ambiguity about domain and geometric interpretation to prevent errors in optimization. "
        "Context: " + taskInfo
    )
    cot_sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking1, answer1 = await cot_sc_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, formulating problem, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings_1 + possible_answers_1, "Sub-task 1: Synthesize and choose the most consistent problem formulation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Derive the system of equations representing the constraints: surface area and volume, "
        "and express the objective function (squared diagonal length) in terms of x, y, z. "
        "Ensure constraints and objective are correctly and clearly formulated to avoid mistakes. "
        "Context: " + answer1.content
    )
    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_sc_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2[i].id}, deriving constraints and objective, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings_2 + possible_answers_2, "Sub-task 2: Synthesize and choose the most consistent constraints and objective.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    debate_instruction_3_1 = (
        "Sub-task 3.1: Apply Lagrange multipliers to find critical points of the squared diagonal length "
        "subject to the two nonlinear constraints (surface area and volume). Carefully handle the system, "
        "respect positivity, and address nonlinear nature without oversimplifications. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer. "
        "Context: " + answer2.content
    )
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_3_1 = []
    all_answer_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_3_1):
        thinking3_1, answer3_1 = await agent([taskInfo, thinking2, answer2], debate_instruction_3_1, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, Lagrange multipliers solving, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
        all_thinking_3_1.append(thinking3_1)
        all_answer_3_1.append(answer3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await final_decision_agent_3_1([taskInfo, thinking2, answer2] + all_thinking_3_1 + all_answer_3_1, "Sub-task 3.1: Synthesize and choose the most consistent Lagrange multiplier solution.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    cot_sc_instruction_3_2 = (
        "Sub-task 3.2: Analyze the system obtained from Lagrange multipliers to find explicit relations "
        "between variables and multipliers. Simplify the system, check consistency and positivity, "
        "and avoid ignoring extraneous or non-physical solutions. "
        "Context: " + answer3_1.content
    )
    cot_sc_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_3_2 = []
    possible_thinkings_3_2 = []
    subtask_desc_3_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_3_2,
        "context": ["user query", thinking3_1.content, answer3_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking3_2, answer3_2 = await cot_sc_agents_3_2[i]([taskInfo, thinking3_1, answer3_1], cot_sc_instruction_3_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3_2[i].id}, analyzing Lagrange system, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
        possible_answers_3_2.append(answer3_2)
        possible_thinkings_3_2.append(thinking3_2)
    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_2, answer3_2 = await final_decision_agent_3_2([taskInfo, thinking3_1, answer3_1] + possible_thinkings_3_2 + possible_answers_3_2, "Sub-task 3.2: Synthesize and choose the most consistent analysis of Lagrange system.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking3_2, "answer": answer3_2}
    logs.append(subtask_desc_3_2)
    print("Step 3.2: ", sub_tasks[-1])

    debate_instruction_4_1 = (
        "Sub-task 4.1: Solve the simplified system to find the maximal squared diagonal length r^2 = d^2/4. "
        "Express the result as a reduced fraction p/q with p, q positive and relatively prime. "
        "Verify the solution satisfies all constraints and positivity conditions. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer. "
        "Context: " + answer3_2.content
    )
    debate_agents_4_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_4_1 = []
    all_answer_4_1 = []
    subtask_desc_4_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_4_1,
        "context": ["user query", thinking3_2.content, answer3_2.content],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_4_1):
        thinking4_1, answer4_1 = await agent([taskInfo, thinking3_2, answer3_2], debate_instruction_4_1, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, solving maximal diagonal, thinking: {thinking4_1.content}; answer: {answer4_1.content}")
        all_thinking_4_1.append(thinking4_1)
        all_answer_4_1.append(answer4_1)
    final_decision_agent_4_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4_1, answer4_1 = await final_decision_agent_4_1([taskInfo, thinking3_2, answer3_2] + all_thinking_4_1 + all_answer_4_1, "Sub-task 4.1: Synthesize and choose the most consistent maximal diagonal solution.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4.1 output: thinking - {thinking4_1.content}; answer - {answer4_1.content}")
    subtask_desc_4_1['response'] = {"thinking": thinking4_1, "answer": answer4_1}
    logs.append(subtask_desc_4_1)
    print("Step 4.1: ", sub_tasks[-1])

    cot_sc_instruction_4_2 = (
        "Sub-task 4.2: Compute and output the final answer p + q, where r^2 = p/q in lowest terms. "
        "Include verification that fraction is reduced and solution corresponds to maximal radius squared. "
        "Context: " + answer4_1.content
    )
    cot_sc_agents_4_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_4_2 = []
    possible_thinkings_4_2 = []
    subtask_desc_4_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": cot_sc_instruction_4_2,
        "context": ["user query", thinking4_1.content, answer4_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking4_2, answer4_2 = await cot_sc_agents_4_2[i]([taskInfo, thinking4_1, answer4_1], cot_sc_instruction_4_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_4_2[i].id}, computing final answer, thinking: {thinking4_2.content}; answer: {answer4_2.content}")
        possible_answers_4_2.append(answer4_2)
        possible_thinkings_4_2.append(thinking4_2)
    final_decision_agent_4_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4_2, answer4_2 = await final_decision_agent_4_2([taskInfo, thinking4_1, answer4_1] + possible_thinkings_4_2 + possible_answers_4_2, "Sub-task 4.2: Synthesize and choose the most consistent final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4.2 output: thinking - {thinking4_2.content}; answer - {answer4_2.content}")
    subtask_desc_4_2['response'] = {"thinking": thinking4_2, "answer": answer4_2}
    logs.append(subtask_desc_4_2)
    print("Step 4.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4_2, answer4_2, sub_tasks, agents)
    return final_answer, logs
