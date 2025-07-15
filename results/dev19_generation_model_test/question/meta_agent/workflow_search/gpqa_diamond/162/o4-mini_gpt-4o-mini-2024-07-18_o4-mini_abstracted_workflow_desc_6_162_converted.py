async def forward_162(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Extract and organize given information (SC_CoT)
    cot_sc_instruction0 = (
        "Sub-task 0: Extract and organize all given quantities and required constants: "
        "mass of Fe(OH)3, total volume, acid concentration, reaction stoichiometry, needed K_sp, "
        "and assumptions about final volume and activity."
    )
    N0 = self.max_sc
    cot_agents0 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                     model=self.node_model, temperature=0.5)
        for _ in range(N0)
    ]
    possible_thinkings0 = []
    possible_answers0 = []
    subtask_desc0 = {
        "subtask_id": "subtask_0_1",
        "instruction": cot_sc_instruction0,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N0):
        thinking0, answer0 = await cot_agents0[i]([taskInfo], cot_sc_instruction0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents0[i].id}, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_thinkings0.append(thinking0)
        possible_answers0.append(answer0)
    final_decision0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                   model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision0(
        [taskInfo] + possible_thinkings0 + possible_answers0,
        "Sub-task 0: Synthesize and choose the most consistent extraction of data.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 1: ", sub_tasks[-1])

    # Stage 1, Subtask 1: Calculate moles of Fe(OH)3 and H+ needed (SC_CoT)
    cot_sc_instruction1 = (
        "Sub-task 1: Calculate the moles of Fe(OH)3 present and determine the moles of H+ "
        "required for complete neutralization."
    )
    N1 = self.max_sc
    cot_agents1 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                     model=self.node_model, temperature=0.5)
        for _ in range(N1)
    ]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1_1",
        "instruction": cot_sc_instruction1,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N1):
        thinking1, answer1 = await cot_agents1[i]([taskInfo, thinking0, answer0], cot_sc_instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thinkings1.append(thinking1)
        possible_answers1.append(answer1)
    final_decision1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                   model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision1(
        [taskInfo, thinking0, answer0] + possible_thinkings1 + possible_answers1,
        "Sub-task 1: Synthesize and choose the most consistent H+ requirement.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 2: ", sub_tasks[-1])

    # Stage 1, Subtask 2: Compute acid volume and excess [H+] (SC_CoT)
    cot_sc_instruction2 = (
        "Sub-task 2: Using the required moles of H+ and the 0.1 M acid, compute the minimum acid "
        "volume to both dissolve Fe(OH)3 and leave an excess [H+] that establishes the final pH, "
        "incorporating the K_sp equilibrium if needed."
    )
    N2 = self.max_sc
    cot_agents2 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                     model=self.node_model, temperature=0.5)
        for _ in range(N2)
    ]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_1_2",
        "instruction": cot_sc_instruction2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2, answer2 = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_thinkings2.append(thinking2)
        possible_answers2.append(answer2)
    final_decision2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                   model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision2(
        [taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2,
        "Sub-task 2: Synthesize and choose the most consistent acid volume and excess [H+].",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2, Subtask 3: Determine pH and select the matching choice (Reflexion)
    reflect_inst = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction = (
        "Sub-task 3: From the calculated excess [H+], determine the pH and compare the computed volume and pH to the provided choices to select the correct answer. "
        + reflect_inst
    )
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                              model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback", "correct"], "Critic Agent",
                                 model=self.node_model, temperature=0.0)
    N_max = self.max_round
    inputs3 = [taskInfo, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_2_1",
        "instruction": cot_reflect_instruction,
        "context": ["user query", thinking2, answer2],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent3(inputs3, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        critic_inst = (
            "Please review the answer above and criticize on where might be wrong. "
            "If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        )
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback3.content}; correct: {correct3.content}")
        if correct3.content == "True":
            break
        inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(inputs3, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refining thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs