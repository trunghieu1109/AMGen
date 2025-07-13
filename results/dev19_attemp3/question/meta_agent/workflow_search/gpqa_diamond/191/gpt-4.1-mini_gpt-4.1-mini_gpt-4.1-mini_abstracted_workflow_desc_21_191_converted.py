async def forward_191(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Analyze and classify the given physical elements and parameters: the spherical conductor, cavity, charge placement, and observation point, "
        "including their geometric and electrostatic relationships. Explicitly identify and clarify all problem parameters and vector definitions to avoid ambiguity in subsequent reasoning."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1 = {
        "subtask_id": "stage1_subtask1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing problem parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    reflect_inst_2 = (
        "Sub-task 2: Apply Gauss's law and the conductor shielding theorem to rigorously determine the nature of the electric field outside the uncharged spherical conductor with an internal cavity containing charge +q. "
        "Explicitly derive and confirm that the induced charges on the conductor's outer surface exactly cancel all multipole moments of the internal charge distribution, resulting in an external field equivalent to that of a point charge +q located at the conductor's center. "
        "This subtask must address and correct the previous failure to incorporate the conductor's shielding effect, preventing any assumptions that the external field depends on the cavity's offset or internal geometry."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2 = self.max_round
    cot_inputs_2 = [taskInfo, thinking1, answer1]
    subtask_desc_2 = {
        "subtask_id": "stage1_subtask2",
        "instruction": reflect_inst_2,
        "context": ["user query", "thinking of stage1_subtask1", "answer of stage1_subtask1"],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot_agent_2(cot_inputs_2, reflect_inst_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, applying conductor shielding, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(N_max_2):
        critic_inst_2 = (
            "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
        )
        feedback, correct = await critic_agent_2([taskInfo, thinking2, answer2], critic_inst_2, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_2(cot_inputs_2, reflect_inst_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining conductor shielding, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_3 = (
        "Sub-task 3: Generate and evaluate possible configurations and candidate expressions of the electric field magnitude at point P outside the conductor, "
        "ensuring all reasoning strictly references and does not contradict the physical principle established in subtask_2. Avoid geometric approximations that treat the charge +q as if it were in free space inside the cavity. "
        "Critically assess each candidate formula in light of the conductor shielding effect and the monopole equivalence principle. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_3 = self.max_round
    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]
    subtask_desc_3 = {
        "subtask_id": "stage1_subtask3",
        "instruction": debate_instr_3,
        "context": ["user query", "thinking of stage1_subtask2", "answer of stage1_subtask2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instr_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2, answer2] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instr_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_3[r].append(thinking3)
            all_answer_3[r].append(answer3)
    final_decision_instr_3 = (
        "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    )
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3(
        [taskInfo, thinking2, answer2] + all_thinking_3[-1] + all_answer_3[-1],
        "Sub-task 3: Evaluate candidate expressions and select the best." + final_decision_instr_3,
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, evaluating candidates, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Synthesize insights from the physical principle verification (subtask_2) and the evaluation of candidate field expressions (stage_1.subtask_3) "
        "to identify and confirm the correct expression for the magnitude of the electric field at point P outside the conductor. Ensure the final conclusion respects Gauss's law and the conductor shielding theorem, explicitly rejecting any formula inconsistent with these principles. Provide a clear, physically justified final answer."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_4 = {
        "subtask_id": "stage2_subtask1",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of stage1_subtask2", "answer of stage1_subtask2", "thinking of stage1_subtask3", "answer of stage1_subtask3"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, synthesizing final answer, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
