async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Formulate a precise mathematical representation of the convex equilateral hexagon ABCDEF with opposite sides parallel. "
        "Define vectors or coordinates for vertices or sides capturing the constraints: all sides equal length, and AB||DE, BC||EF, CD||FA. "
        "Avoid assuming numeric values for side length."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formulating hexagon representation, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Represent the triangle formed by the extensions of sides AB, CD, and EF. "
        "Identify intersection points of these extended lines and express the triangle's side lengths in terms of the hexagon's side length and the vectors or coordinates from Sub-task 1. "
        "Consider orientation and relative positions carefully without assuming the triangle's location."
    )
    N_sc = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, representing triangle from extended sides, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_answers_0_2 + possible_thinkings_0_2, "Sub-task 2: Synthesize and choose the most consistent triangle representation from extended sides.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_0_2.id}, synthesizing triangle representation, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    cot_reflect_instruction_0_3 = (
        "Sub-task 3: Validate the derived representations from Sub-tasks 1 and 2 by checking consistency with the problem's conditions: convexity, equilateral sides, parallel opposite sides, and the triangle side lengths 200, 240, 300. "
        "Adjust the model if inconsistencies arise. Avoid skipping validation."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_0_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_0_3 = self.max_round
    cot_inputs_0_3 = [taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2]
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_reflect_instruction_0_3,
        "context": ["user query", thinking_0_1.content, answer_0_1.content, thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3(cot_inputs_0_3, cot_reflect_instruction_0_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_0_3.id}, validating representations, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    for i in range(N_max_0_3):
        feedback, correct = await critic_agent_0_3([taskInfo, thinking_0_3, answer_0_3], "Please review and provide limitations of the solution. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_0_3.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_0_3.extend([thinking_0_3, answer_0_3, feedback])
        thinking_0_3, answer_0_3 = await cot_agent_0_3(cot_inputs_0_3, cot_reflect_instruction_0_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_0_3.id}, refining validation, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_1_1 = (
        "Sub-task 1: Using the validated representations from Stage 0, derive algebraic equations relating the hexagon's side length to the triangle's known side lengths 200, 240, and 300. "
        "Express the triangle's side lengths as functions of the hexagon's side length and solve the system. Emphasize careful algebraic manipulation and geometric interpretation."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_3, answer_0_3], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, deriving algebraic relations, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Solve the system of equations from Sub-task 1 to find the numerical value of the hexagon's side length. "
        "Verify the solution's consistency with all geometric constraints and the given triangle side lengths. Avoid accepting invalid solutions."
    )
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, solving equations, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent solution for hexagon side length.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_1_2.id}, synthesizing solution, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 5: ", sub_tasks[-1])

    cot_reflect_instruction_1_3 = (
        "Sub-task 3: Provide the final answer for the hexagon's side length along with a verification summary. "
        "Explain how the solution satisfies all problem conditions and check for ambiguities or alternative solutions. "
        "If verification fails, revisit previous subtasks."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_3 = self.max_round
    cot_inputs_1_3 = [taskInfo, thinking_1_2, answer_1_2]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_reflect_instruction_1_3,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, verifying final answer, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    for i in range(N_max_1_3):
        feedback, correct = await critic_agent_1_3([taskInfo, thinking_1_3, answer_1_3], "Please review and provide limitations of the final answer. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_3.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1_3.extend([thinking_1_3, answer_1_3, feedback])
        thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, refining verification, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_1_3, answer_1_3, sub_tasks, agents)
    return final_answer, logs
