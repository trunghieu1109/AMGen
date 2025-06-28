async def forward_8(self, taskInfo):
    from collections import Counter

    print("Task Requirement: ", taskInfo)

    sub_tasks = []
    agents = []

    # Sub-task 1: Gather 3D structural representations
    cot_instruction = "Sub-task 1: Gather reliable three-dimensional structural representations (CIF files, 3D models, or accurate drawings) for each of the four molecules: triisopropyl borate, quinuclidine, benzo[1,2-c:3,4-c:5,6-c]trifuran-1,3,4,6,7,9-hexaone, and triphenyleno[1,2-c:5,6-c:9,10-c]trifuran-1,3,6,8,11,13-hexaone."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, gathering structures, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Identify symmetry elements in triisopropyl borate
    cot_instruction2 = "Sub-task 2: Using the structural data from Sub-task 1, identify and catalog all symmetry elements present in triisopropyl borate."
    cot_agent2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent2([taskInfo, thinking1, answer1], cot_instruction2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2.id}, identifying symmetry in triisopropyl borate, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Identify symmetry elements in quinuclidine
    cot_instruction3 = "Sub-task 3: Using the structural data from Sub-task 1, identify and catalog all symmetry elements present in quinuclidine."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, identifying symmetry in quinuclidine, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Identify symmetry elements in benzo-trifuran hexaone
    cot_instruction4 = "Sub-task 4: Using the structural data from Sub-task 1, identify and catalog all symmetry elements present in benzo[1,2-c:3,4-c:5,6-c]trifuran-1,3,4,6,7,9-hexaone."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, thinking1, answer1], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, identifying symmetry in benzo-trifuran hexaone, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Identify symmetry elements in triphenyleno-trifuran hexaone
    cot_instruction5 = "Sub-task 5: Using the structural data from Sub-task 1, identify and catalog all symmetry elements present in triphenyleno[1,2-c:5,6-c:9,10-c]trifuran-1,3,6,8,11,13-hexaone."
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent5([taskInfo, thinking1, answer1], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, identifying symmetry in triphenyleno-trifuran hexaone, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Evaluate C3h symmetry criteria
    reflect_instruction6 = "Sub-task 6: Evaluate each molecule’s catalog of symmetry elements from Sub-tasks 2–5 to determine whether it possesses both a principal C3 rotation axis and a horizontal mirror plane (σh), defining C3h point-group symmetry."
    cot_agent6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs6 = [taskInfo, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5]
    thinking6, answer6 = await cot_agent6(cot_inputs6, reflect_instruction6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent6.id}, evaluating C3h criteria, thinking: {thinking6.content}; answer: {answer6.content}")
    N_max = self.max_round
    for i in range(N_max):
        feedback6, correct6 = await critic_agent6([taskInfo, thinking6, answer6], "Review the evaluation for completeness and correctness; indicate if both C3 axis and σh plane are correctly identified.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent6.id}, providing feedback, thinking: {feedback6.content}; answer: {correct6.content}")
        if correct6.content == "True":
            break
        cot_inputs6.extend([thinking6, answer6, feedback6])
        thinking6, answer6 = await cot_agent6(cot_inputs6, reflect_instruction6, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent6.id}, refining C3h evaluation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    # Sub-task 7: Select molecules satisfying C3h symmetry
    debate_instruction7 = "Sub-task 7: Based on the evaluation in Sub-task 6, select and report the molecule or molecules that satisfy C3h point-group symmetry."
    debate_agents7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking7 = []
    all_answer7 = []
    for agent in debate_agents7:
        thinking7, answer7 = await agent([taskInfo, thinking6, answer6], debate_instruction7, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, proposing selection, thinking: {thinking7.content}; answer: {answer7.content}")
        all_thinking7.append(thinking7)
        all_answer7.append(answer7)
    final_decision_agent7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent7([taskInfo] + all_thinking7 + all_answer7, "Sub-task 7: Make final decision on which molecule(s) have C3h symmetry.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent7.id}, deciding final molecules, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Subtask 7 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer