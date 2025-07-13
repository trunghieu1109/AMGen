async def forward_8(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_stage0 = (
        "Sub-task 1: Derive a formal representation of the game states for the game where players remove 1 or 4 tokens alternately, "
        "starting with Alice. Define each state as the number of tokens remaining and classify it as winning or losing for the player about to move. "
        "Use the rules: a position is losing if all moves lead to winning positions, and winning if at least one move leads to a losing position. "
        "Validate this representation by checking base cases such as n=0 and small values of n."
    )
    cot_agent_stage0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_stage0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking0, answer0 = await cot_agent_stage0([taskInfo], cot_instruction_stage0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_stage0.id}, deriving and validating game state representation, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {
        "thinking": thinking0,
        "answer": answer0
    }
    logs.append(subtask_desc0)

    cot_sc_instruction_stage1 = (
        "Sub-task 2: Using the representation from Sub-task 1, compute and enumerate the classification (winning or losing) for all positions n from 1 to 2024. "
        "Identify losing positions for the first player (Alice), which correspond to positions where Bob can guarantee a win. "
        "Verify correctness by cross-checking with known small cases and ensuring consistency with the recursive definition."
    )
    N_sc = self.max_sc
    cot_agents_stage1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_stage1 = []
    possible_thinkings_stage1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_stage1,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking1, answer1 = await cot_agents_stage1[i]([taskInfo, thinking0, answer0], cot_sc_instruction_stage1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage1[i].id}, enumerating and verifying positions 1 to 2024, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_stage1.append(answer1)
        possible_thinkings_stage1.append(thinking1)

    final_decision_agent_stage1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_final, answer1_final = await final_decision_agent_stage1(
        [taskInfo] + possible_answers_stage1 + possible_thinkings_stage1,
        "Sub-task 2: Synthesize and choose the most consistent and correct classification of positions 1 to 2024.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1_final.content}; answer - {answer1_final.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1_final,
        "answer": answer1_final
    }
    logs.append(subtask_desc1)

    cot_reflect_instruction_stage2 = (
        "Sub-task 3: Aggregate the classification results from Sub-task 2 to count how many positive integers n ≤ 2024 are losing positions for Alice, "
        "meaning Bob has a guaranteed winning strategy. Verify the final count by reasoning about any pattern or periodicity in the classification to ensure correctness and efficiency. "
        "Use insights from previous subtasks and feedback to improve accuracy."
    )
    cot_agent_stage2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_stage2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_reflect = self.max_round
    cot_inputs_stage2 = [taskInfo, thinking0, answer0, thinking1_final, answer1_final]
    subtask_desc2 = {
        "subtask_id": "subtask_1",
        "instruction": cot_reflect_instruction_stage2,
        "context": ["user query", "thinking and answer of subtask 1", "thinking and answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot_agent_stage2(cot_inputs_stage2, cot_reflect_instruction_stage2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_stage2.id}, aggregating and verifying final count, thinking: {thinking2.content}; answer: {answer2.content}")

    for i in range(N_max_reflect):
        feedback, correct = await critic_agent_stage2(
            [taskInfo, thinking2, answer2],
            "Please review the answer above and criticize any possible errors. If absolutely correct, output exactly 'True' in 'correct'.",
            i,
            is_sub_task=True
        )
        agents.append(f"Critic agent {critic_agent_stage2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_stage2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_stage2(cot_inputs_stage2, cot_reflect_instruction_stage2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_stage2.id}, refining final count, thinking: {thinking2.content}; answer: {answer2.content}")

    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)

    final_answer = await self.make_final_answer(thinking2, answer2, sub_tasks, agents)
    for i, step in enumerate(sub_tasks, 1):
        print(f"Step {i}: ", step)
    return final_answer, logs
